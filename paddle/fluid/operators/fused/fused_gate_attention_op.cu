/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class FusedGateAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto *input_x = ctx.Input<Tensor>("X");
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    // std::cout << "\ne23e3ererew\n";
    auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *gate_weight = ctx.Input<Tensor>("GateWeight");
    auto *gate_bias = ctx.Input<Tensor>("GateBias");
    auto *out_linear_weight = ctx.Input<Tensor>("OutLinearW");
    auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");

    auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");
    
    const auto is_gating = ctx.Attr<bool>("is_gating");

    auto *qkv_out = ctx.Output<Tensor>("QKVOut");
    auto *transpose_out_2 = ctx.Output<Tensor>("TransposeOut2");
    auto *qk_out = ctx.Output<Tensor>("QKOut");
    auto *qktv_out = ctx.Output<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *src_mask_out = ctx.Output<Tensor>("SrcMaskOut");
    auto *fmha_out = ctx.Output<Tensor>("FMHAOut");
    auto *out_linear_out = ctx.Output<Tensor>("OutLinearOut");
    auto *gate_value_out = ctx.Output<Tensor>("GateValueOut");
    auto *gate_bias_out = ctx.Output<Tensor>("GateBiasOut");
    auto *gate_out = ctx.Output<Tensor>("GateOut");

    qkv_out->mutable_data<T>(ctx.GetPlace());
    transpose_out_2->mutable_data<T>(ctx.GetPlace());
    qk_out->mutable_data<T>(ctx.GetPlace());
    qktv_out->mutable_data<T>(ctx.GetPlace());
    softmax_out->mutable_data<T>(ctx.GetPlace());
    src_mask_out->mutable_data<T>(ctx.GetPlace());
    fmha_out->mutable_data<T>(ctx.GetPlace());
    out_linear_out->mutable_data<T>(ctx.GetPlace());

    if(is_gating) {
        gate_value_out->mutable_data<T>(ctx.GetPlace());
        gate_bias_out->mutable_data<T>(ctx.GetPlace());
        gate_out->mutable_data<T>(ctx.GetPlace());
    }

    // final output.
    auto *out = ctx.Output<Tensor>("Y");
    out->mutable_data<T>(ctx.GetPlace());

    // // get data ptr for qkv part.
    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    int batch_size = input_x_dims[0];
    int seq_len_m = input_x_dims[1];
    int seq_len_r = input_x_dims[2];
    int hidden_size = input_x_dims[3];

    // std::cout << "seq_len_m: " << seq_len_m << "\n";

    int num_head = qkv_w_dims[2];
    int c = qkv_w_dims[3];

    auto *x_data = input_x->data<T>();
    auto *qkv_weight_data = qkv_weight->data<T>();
    // Tensor qkv_out;
    // qkv_out.Resize({ batch_size, seq_len_m, seq_len_r, 3, num_head, c});
   

    // int bsz_seq = batch_size * max_seq_len;
    // int hidden_size = num_head * dim_head;
    // (m, n, k) = bsz_seq, output_size, input_size
    int k = hidden_size;
    //nbhqk,nbkhc->nbqhc [batch_size * seq_len_m * seq_len_r * 3 * num_head * c]
    int m = batch_size * seq_len_m * seq_len_r;
    int n = 3 * num_head * c;

    auto qkv_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, m,
                      n, k, false);
    qkv_compute.ComputeForward(qkv_weight, input_x, nullptr, qkv_out, nullptr);
    auto fmha_ref_compute =
        FMHAGateRef<T>(ctx.cuda_device_context(), batch_size, seq_len_m, seq_len_r, num_head, c);

    fmha_ref_compute.ComputeForward(
        nonbatched_bias, *qkv_out, src_mask, transpose_out_2, qk_out,
        src_mask_out, softmax_out,
        qktv_out, fmha_out);

    if(is_gating){
        n = num_head * c;
        auto gate_attn_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, m,
                      n, k, true);
        gate_attn_compute.ComputeForward(gate_weight, input_x, gate_bias, gate_value_out,
        gate_bias_out);
        auto gate_compute = GateRef<T>(ctx);
        gate_compute.ComputeForward(*gate_bias_out, *fmha_out, gate_out);

        fmha_out->ShareDataWith(*gate_out);
    }

    // std::cout << "fmha_out: " << *fmha_out << "\n";
    m = batch_size * seq_len_m * seq_len_r;
    k = num_head * c;
    n = hidden_size;

    // [1,3,2,8,2] * [8,2,4] = [[1, 3, 2, 4]]
    auto out_linear_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, m,
                      n, k, true);

    // std::cout << "out_linear_weight: " << *out_linear_weight << "\n";
    // std::cout << "fmha_out: " << fmha_out << "\n";
    // std::cout << "out_linear_bias: " << out_linear_bias << "\n";
    out_linear_compute.ComputeForward(out_linear_weight, fmha_out, out_linear_bias,
        out_linear_out, out);
    // std::cout << "out: " << *out << "\n";
    // std::cout << "out_linear_out: " << *out_linear_out << "\n";
    // tensor model parallel
    // AllReduce<T>(*out_linear_out, ring_id, ctx.cuda_device_context());
  }
};

template <typename T>
class FusedGateAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;

    std::cout << "\n>>>>>>>>>>>>>>grad>>>>>>>>>>>>>>\n";
    const auto is_gating = ctx.Attr<bool>("is_gating");

    // get inputs.
    auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *d_y_data = d_y->data<T>();
    // fw input
    auto *input_x = ctx.Input<Tensor>("X");
    auto *x_data = input_x->data<T>();
    // auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    // auto *ln_2_scale_data =
        // (ln_2_scale == nullptr ? nullptr : ln_2_scale->data<U>());
    // fw parameters.
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
    auto *gate_weight = ctx.Input<Tensor>("GateWeight");
    // auto *qkv_bias = ctx.Input<Tensor>("QKVBias");
    auto *out_linear_weight = ctx.Input<Tensor>("OutLinearW");
    auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");
    auto *gate_bias = ctx.Input<Tensor>("GateBias");
    auto *src_mask_data = (src_mask == nullptr ? nullptr : src_mask->data<T>());
    auto *qkv_weight_data = qkv_weight->data<T>();
    // auto *qkv_bias_data = (qkv_bias == nullptr) ? nullptr : qkv_bias->data<T>();
    auto *out_linear_weight_data = out_linear_weight->data<T>();
    auto *out_linear_bias_data =
        (out_linear_bias == nullptr) ? nullptr : out_linear_bias->data<T>();

    // fw output
    auto *fmha_out = ctx.Input<Tensor>("FMHAOut");
    auto *gate_bias_out = ctx.Input<Tensor>("GateBiasOut");
    auto *gate_out = ctx.Input<Tensor>("GateOut");
    auto *transpose_out_2 = ctx.Input<Tensor>("TransposeOut2");
    auto *qk_out = ctx.Input<Tensor>("QKOut");
    auto *qktv_out = ctx.Input<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");

    auto *src_mask_out = ctx.Input<Tensor>("SrcMaskOut");
    auto *out_linear_out = ctx.Input<Tensor>("OutLinearOut");

    // auto *bias_dropout_residual_out =
    //     ctx.Input<Tensor>("BiasDropoutResidualOut");
    // auto *fmha_out_data = fmha_out->data<T>();
    // auto *transpose_out_2_data = transpose_out_2->data<T>();
    // auto *qk_out_data = qk_out->data<T>();
    // auto *qktv_out_data = qktv_out->data<T>();
    // auto *softmax_out_data = softmax_out->data<T>();
    auto *src_mask_out_data = src_mask_out->data<T>();
    auto *out_linear_out_data = out_linear_out->data<T>();
    // auto *dropout_mask_out_data = dropout_mask_out->data<uint8_t>();

    // output's grad
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_qkv_out = ctx.Output<Tensor>(framework::GradVarName("QKVOut"));
    auto *d_qktv_out = ctx.Output<Tensor>(framework::GradVarName("QKTVOut"));
    auto *d_transpose_out_2 =
        ctx.Output<Tensor>(framework::GradVarName("TransposeOut2"));
    auto *d_qk_out = ctx.Output<Tensor>(framework::GradVarName("QKOut"));
    auto *d_gate_bias_out = ctx.Output<Tensor>(framework::GradVarName("GateBiasOut"));
    auto *d_gate_out = ctx.Output<Tensor>(framework::GradVarName("GateOut"));
    auto *d_softmax_out =
        ctx.Output<Tensor>(framework::GradVarName("SoftmaxOut"));
    auto *d_src_mask_out =
        ctx.Output<Tensor>(framework::GradVarName("SrcMaskOut"));
    auto *d_fmha_out = ctx.Output<Tensor>(framework::GradVarName("FMHAOut"));
    auto *d_out_linear_out =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearOut"));
    // auto *d_bias_dropout_residual_out =
    //     ctx.Output<Tensor>(framework::GradVarName("BiasDropoutResidualOut"));
    auto *d_x_data = d_x->mutable_data<T>(ctx.GetPlace());
    // when qkv_bias is not nullptr, d_qkv_out is equals to d_qkv_bias_out, the
    // space can be reused.
    auto *d_qkv_out_data = d_qkv_out->mutable_data<T>(ctx.GetPlace());
    // auto *d_qkv_bias_out_data =
    //     (d_qkv_bias_out == nullptr)
    //         ? nullptr
    //         : d_qkv_bias_out->mutable_data<T>(ctx.GetPlace());
    auto *d_qktv_out_data = d_qktv_out->mutable_data<T>(ctx.GetPlace());
    auto *d_transpose_out_2_data =
        d_transpose_out_2->mutable_data<T>(ctx.GetPlace());
    auto *d_qk_out_data = d_qk_out->mutable_data<T>(ctx.GetPlace());
    d_gate_bias_out->mutable_data<T>(ctx.GetPlace());
    d_gate_out->mutable_data<T>(ctx.GetPlace());
    auto *d_softmax_out_data = d_softmax_out->mutable_data<T>(ctx.GetPlace());
    auto *d_src_mask_out_data = d_src_mask_out->mutable_data<T>(ctx.GetPlace());
    auto *d_fmha_out_data = d_fmha_out->mutable_data<T>(ctx.GetPlace());
    auto *d_out_linear_out_data =
        d_out_linear_out->mutable_data<T>(ctx.GetPlace());

    // parameter grad
    auto *d_qkv_weight = ctx.Output<Tensor>(framework::GradVarName("QKVWeight"));
    auto *d_out_linear_weight =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearW"));
    auto *d_gate_weight = ctx.Output<Tensor>(framework::GradVarName("GateWeight"));
    auto *d_out_linear_bias =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));
    auto *d_gate_bias = ctx.Output<Tensor>(framework::GradVarName("GateBias"));

    auto *intermediate_out = ctx.Input<Tensor>("IntermediateOut"); 
    auto *d_intermediate_out =
        ctx.Output<Tensor>(framework::GradVarName("IntermediateOut"));

    auto *d_qkv_weight_data = d_qkv_weight->mutable_data<T>(ctx.GetPlace());
    auto *d_out_linear_weight_data =
        d_out_linear_weight->mutable_data<T>(ctx.GetPlace());
    d_gate_weight->mutable_data<T>(ctx.GetPlace());
    auto *d_out_linear_bias_data = d_out_linear_bias->mutable_data<T>(ctx.GetPlace());
    d_gate_bias->mutable_data<T>(ctx.GetPlace());

    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    int batch_size = input_x_dims[0];
    int seq_len_m = input_x_dims[1];
    int seq_len_r = input_x_dims[2];
    int hidden_size = input_x_dims[3];

    int num_head = qkv_w_dims[2];
    int c = qkv_w_dims[3];

    Tensor d_residual;
    d_residual.Resize(input_x_dims);
    T *d_residual_data = d_residual.mutable_data<T>(ctx.GetPlace());

    bool transA = false;
    bool transB = true;

    int k = hidden_size;
    //nbhqk,nbkhc->nbqhc [batch_size * seq_len_m * seq_len_r * 3 * num_head * c]
    int m = batch_size * seq_len_m * seq_len_r;
    int n = 3 * num_head * c;
    auto qkv_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), transA, transB, m,
                      n, k, false);

    auto fmha_ref_compute =
        FMHAGateRef<T>(ctx.cuda_device_context(), batch_size, seq_len_m, seq_len_r, num_head,
                   c);
    
    m = batch_size * seq_len_m * seq_len_r;
    k = num_head * c;
    n = hidden_size;

    // (b*s, num_head * dim_head) * (num_head * dim_head, dim_embed)
    auto out_linear_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, m,
                      n, k, true);
    std::cout << "fmha_out: " << *fmha_out << "\n";
    std::cout << "out_linear_weight: " << *out_linear_weight << "\n";
    std::cout << "d_y: " << *d_y << "\n";
    std::cout << "d_fmha_out: " << *d_fmha_out << "\n";
    std::cout << "d_out_linear_weight: " << *d_out_linear_weight << "\n";
    std::cout << "d_out_linear_bias: " << *d_out_linear_bias << "\n";
    // std::cout << "d_out_linear_out: " << *d_out_linear_out << "\n";
    // std::cout << "d_out_linear_bias: " << *d_out_linear_bias << "\n";
    out_linear_compute.ComputeBackward(fmha_out, out_linear_weight,
        d_y, d_fmha_out,
        d_out_linear_weight, d_out_linear_bias);
   

    if(is_gating){
        d_gate_out->ShareDataWith(*d_fmha_out);
        std::cout << "\n>>>>>>>>>>>is_gating>>>>>>>>>>>\n";
        auto gate_compute = GateRef<T>(ctx);
        // gate_compute.ComputeForward(*gate_bias_out, *fmha_out, sigmoid_out, gate_out);
        gate_compute.ComputeBackward(fmha_out, gate_bias_out, gate_out, d_gate_out, intermediate_out, d_fmha_out, d_gate_bias_out, d_intermediate_out);
        std::cout << "d_gate_bias_out: " << *d_gate_bias_out << "\n";
        n = num_head * c;
        auto gate_attn_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, m,
                      n, k, true);
        gate_attn_compute.ComputeBackward(input_x, gate_weight, d_gate_bias_out, d_x, d_gate_weight, d_gate_bias);
        
        std::cout << "d_gate_bias: " << d_gate_bias << "\n";
    }

    // fmha_ref_compute.ComputeBackward(
    //     *transpose_out_2, src_mask, *softmax_out,
    //      *qk_out, *src_mask_out, *d_fmha_out, d_qktv_out,
    //     d_softmax_out, d_src_mask_out, d_qk_out,
    //     d_transpose_out_2, nullptr, d_qkv_out);

    // qkv_compute.ComputeBackward(input_x, qkv_weight, d_qkv_out, d_x,
    //                                 d_qkv_weight, d_qkv_bias);
      
    // //   // tensor model parallel
    // //   AllReduce<T>(*d_x, ring_id, ctx.cuda_device_context());

    // // gradient accumulation
    // std::vector<const Tensor *> ins;
    // std::vector<Tensor *> outs;
    // ins.emplace_back(&d_residual);
    // ins.emplace_back(d_x);
    // outs.emplace_back(d_x);
    // int elewise_add_axis = -1;
    // paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
    //                                                T>(
    //     ctx.cuda_device_context(), ins, &outs, elewise_add_axis,
    //     AddFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_gate_attention, ops::FusedGateAttentionOpKernel<float>,
                        ops::FusedGateAttentionOpKernel<double>,
                        ops::FusedGateAttentionOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(fused_gate_attention_grad,
                        ops::FusedGateAttentionGradKernel<float>,
                        ops::FusedGateAttentionGradKernel<double>,
                        ops::FusedGateAttentionGradKernel<plat::float16>);
