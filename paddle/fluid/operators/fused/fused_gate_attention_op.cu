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
#include "paddle/fluid/memory/memory.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
Tensor *ComputeMergedQKVMatmulForward(const framework::ExecutionContext &ctx,
                                      const Tensor *x, int m, int n, int k) {
  // LOG(INFO) << "Compute Merged QKV Matmul Forward";
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
  auto *qkv_out = ctx.Output<Tensor>("QKVOut");
  qkv_out->mutable_data<T>(ctx.GetPlace());

  // qkv_out = GEMM(x, qkv_weight)
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeForward(qkv_weight, x, nullptr, qkv_out, nullptr);
  return qkv_out;
}

template <typename T>
Tensor *ComputeMergedQKVMatmulBackward(const framework::ExecutionContext &ctx,
                                       const Tensor *x, const Tensor *d_qkv_out,
                                       Tensor *d_x, int m, int n, int k) {
  auto *qkv_weight = ctx.Input<Tensor>("QKVWeight");
  auto *d_qkv_weight = ctx.Output<Tensor>(framework::GradVarName("QKVWeight"));
  d_qkv_weight->mutable_data<T>(ctx.GetPlace());

  // Gradient of GEMM(x, qkv_weight)
  auto qkv_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, true, m, n, k, false);
  qkv_compute.ComputeBackward(x, qkv_weight, d_qkv_out, d_x, d_qkv_weight,
                              nullptr);
  return d_x;
}

template <typename T>
Tensor *ComputeGatingLinearForward(const framework::ExecutionContext &ctx,
                                   const Tensor *x, const Tensor *fmha_out,
                                   int m, int n, int k) {
  // LOG(INFO) << "Compute Gating Linear Forward";
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");
  auto *gate_bias = ctx.Input<Tensor>("GateBias");

  auto *gate_bias_out = ctx.Output<Tensor>("GateBiasOut");
  auto *sigmoid_out = ctx.Output<Tensor>("SigmoidOut");
  auto *gate_out = ctx.Output<Tensor>("GateOut");

  gate_bias_out->mutable_data<T>(ctx.GetPlace());
  sigmoid_out->mutable_data<T>(ctx.GetPlace());
  gate_out->mutable_data<T>(ctx.GetPlace());

  // The first gate_bias_out stores the result of the multiplication,
  // and the second gate_bias_out stores the result of the multiplication +
  // bias.
  //   gate_bias_out = GEMM(input_x, gate_weight)
  //   gate_bias_out = gate_bias_out + gate_bias
  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeForward(gate_weight, x, gate_bias, gate_bias_out,
                                   gate_bias_out);

  // sigmoid_out = sigmoid(gate_bias_out)
  // gate_out = sigmoid_out * fmha_out
  auto gate_compute = GateRef<T>(ctx.cuda_device_context());
  gate_compute.ComputeForward(*gate_bias_out, *fmha_out, sigmoid_out, gate_out);
  return gate_out;
}

template <typename T>
Tensor *ComputeGatingLinearBackward(const framework::ExecutionContext &ctx,
                                    const Tensor *d_gate_out, int m, int n,
                                    int k) {
  auto *fmha_out = ctx.Input<Tensor>("FMHAOut");
  auto *gate_bias_out = ctx.Input<Tensor>("GateBiasOut");
  auto *gate_out = ctx.Input<Tensor>("GateOut");
  auto *sigmoid_out = ctx.Input<Tensor>("SigmoidOut");

  auto *input_x = ctx.Input<Tensor>("X");
  auto *gate_weight = ctx.Input<Tensor>("GateWeight");

  auto *d_fmha_out = ctx.Output<Tensor>(framework::GradVarName("FMHAOut"));
  auto *d_gate_bias_out =
      ctx.Output<Tensor>(framework::GradVarName("GateBiasOut"));
  auto *d_sigmoid_out =
      ctx.Output<Tensor>(framework::GradVarName("SigmoidOut"));

  d_fmha_out->mutable_data<T>(ctx.GetPlace());
  d_gate_bias_out->mutable_data<T>(ctx.GetPlace());
  d_sigmoid_out->mutable_data<T>(ctx.GetPlace());

  auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
  auto *d_gate_weight =
      ctx.Output<Tensor>(framework::GradVarName("GateWeight"));
  auto *d_gate_bias = ctx.Output<Tensor>(framework::GradVarName("GateBias"));

  d_x->mutable_data<T>(ctx.GetPlace());
  d_gate_weight->mutable_data<T>(ctx.GetPlace());
  d_gate_bias->mutable_data<T>(ctx.GetPlace());

  // Gradient of sigmoid(gate_bias_out) * fmha_out
  auto gate_compute = GateRef<T>(ctx.cuda_device_context());
  gate_compute.ComputeBackward(*fmha_out, *gate_bias_out, *gate_out,
                               *d_gate_out, *sigmoid_out, d_fmha_out,
                               d_gate_bias_out, d_sigmoid_out);

  auto gate_attn_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  gate_attn_compute.ComputeBackward(input_x, gate_weight, d_gate_bias_out, d_x,
                                    d_gate_weight, d_gate_bias);
  return d_fmha_out;
}

template <typename T>
Tensor *ComputeOutputLinearForward(const framework::ExecutionContext &ctx,
                                   const Tensor *gate_out, int m, int n,
                                   int k) {
  // LOG(INFO) << "Compute Output Linear Forward";
  auto *out_linear_weight = ctx.Input<Tensor>("OutLinearW");
  auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");

  auto *out = ctx.Output<Tensor>("Y");
  out->mutable_data<T>(ctx.GetPlace());

  // out = GEMM(gate_out, out_linear_weight)
  // out = out + out_linear_bias
  auto out_linear_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear_compute.ComputeForward(out_linear_weight, gate_out,
                                    out_linear_bias, out, out);
  return out;
}

template <typename T>
Tensor *ComputeOutputLinearBackward(const framework::ExecutionContext &ctx,
                                    int m, int n, int k, bool is_gating) {
  std::string input_name = is_gating ? "GateOut" : "FMHAOut";

  auto *d_out = ctx.Input<Tensor>(framework::GradVarName("Y"));
  auto *out_linear_weight = ctx.Input<Tensor>("OutLinearW");
  auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");
  auto *input = ctx.Input<Tensor>(input_name);

  auto *d_out_linear_weight =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearW"));
  auto *d_out_linear_bias =
      ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));
  auto *d_input = ctx.Output<Tensor>(framework::GradVarName(input_name));

  d_out_linear_weight->mutable_data<T>(ctx.GetPlace());
  d_out_linear_bias->mutable_data<T>(ctx.GetPlace());
  d_input->mutable_data<T>(ctx.GetPlace());

  auto out_linear_compute =
      AttnMatMul<T>(ctx.cuda_device_context(), false, false, m, n, k, true);
  out_linear_compute.ComputeBackward(input, out_linear_weight, d_out, d_input,
                                     d_out_linear_weight, d_out_linear_bias);
  return d_input;
}

template <typename T>
class FusedGateAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto *input_x = ctx.Input<Tensor>("X");
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto *src_mask = ctx.Input<Tensor>("SrcMask");

    auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");

    const auto is_gating = ctx.Attr<bool>("is_gating");

    auto *transpose_out_2 = ctx.Output<Tensor>("TransposeOut2");
    auto *qk_out = ctx.Output<Tensor>("QKOut");
    auto *qktv_out = ctx.Output<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *src_mask_out = ctx.Output<Tensor>("SrcMaskOut");
    auto *fmha_out = ctx.Output<Tensor>("FMHAOut");

    transpose_out_2->mutable_data<T>(ctx.GetPlace());
    qk_out->mutable_data<T>(ctx.GetPlace());
    qktv_out->mutable_data<T>(ctx.GetPlace());
    softmax_out->mutable_data<T>(ctx.GetPlace());
    src_mask_out->mutable_data<T>(ctx.GetPlace());
    fmha_out->mutable_data<T>(ctx.GetPlace());

    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = ctx.Input<Tensor>("QKVWeight")->dims();
    // LOG(INFO) << "input_x_dims=[" << input_x_dims << "], qkv_w_dims=[" <<
    // qkv_w_dims << "]";

    int batch_size = input_x_dims[0];
    int seq_len_m = input_x_dims[1];
    int seq_len_r = input_x_dims[2];
    int hidden_size = input_x_dims[3];  // qkv_dim

    // qkv_weight[3, n_head, c, qkv_dim]
    int num_head = qkv_w_dims[1];
    int c = qkv_w_dims[2];

    // Merged QKV Matmul
    // nbhqk,nbkhc -> nbqhc
    // [batch_size * seq_len_m * seq_len_r * 3 * num_head * c]
    int m = batch_size * seq_len_m * seq_len_r;
    int n = 3 * num_head * c;
    int k = hidden_size;
    Tensor *qkv_out = ComputeMergedQKVMatmulForward<T>(ctx, input_x, m, n, k);

    auto fmha_ref_compute =
        FMHAGateRef<T>(ctx.cuda_device_context(), batch_size, seq_len_m,
                       seq_len_r, num_head, c);
    fmha_ref_compute.ComputeForward(nonbatched_bias, *qkv_out, src_mask,
                                    transpose_out_2, qk_out, src_mask_out,
                                    softmax_out, qktv_out, fmha_out);

    // Gating Linear
    Tensor *gate_out = nullptr;
    if (is_gating) {
      m = batch_size * seq_len_m * seq_len_r;
      n = num_head * c;
      k = hidden_size;
      gate_out = ComputeGatingLinearForward<T>(ctx, input_x, fmha_out, m, n, k);
    } else {
      gate_out = fmha_out;
    }

    // Output Linear
    m = batch_size * seq_len_m * seq_len_r;
    n = hidden_size;
    k = num_head * c;
    ComputeOutputLinearForward<T>(ctx, gate_out, m, n, k);
  }
};

template <typename T>
class FusedGateAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto is_gating = ctx.Attr<bool>("is_gating");

    // fw input
    auto *input_x = ctx.Input<Tensor>("X");
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *nonbatched_bias = ctx.Input<Tensor>("NonbatchedBias");

    // fw output
    auto *transpose_out_2 = ctx.Input<Tensor>("TransposeOut2");
    auto *qk_out = ctx.Input<Tensor>("QKOut");
    auto *qktv_out = ctx.Input<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");

    auto *src_mask_out = ctx.Input<Tensor>("SrcMaskOut");
    auto *src_mask_out_data = src_mask_out->data<T>();

    // output's grad
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_qkv_out = ctx.Output<Tensor>(framework::GradVarName("QKVOut"));
    auto *d_qktv_out = ctx.Output<Tensor>(framework::GradVarName("QKTVOut"));
    auto *d_transpose_out_2 =
        ctx.Output<Tensor>(framework::GradVarName("TransposeOut2"));
    auto *d_qk_out = ctx.Output<Tensor>(framework::GradVarName("QKOut"));
    auto *d_softmax_out =
        ctx.Output<Tensor>(framework::GradVarName("SoftmaxOut"));
    auto *d_src_mask_out =
        ctx.Output<Tensor>(framework::GradVarName("SrcMaskOut"));

    auto *d_qkv_out_data = d_qkv_out->mutable_data<T>(ctx.GetPlace());
    auto *d_qktv_out_data = d_qktv_out->mutable_data<T>(ctx.GetPlace());
    auto *d_transpose_out_2_data =
        d_transpose_out_2->mutable_data<T>(ctx.GetPlace());
    auto *d_qk_out_data = d_qk_out->mutable_data<T>(ctx.GetPlace());

    auto *d_softmax_out_data = d_softmax_out->mutable_data<T>(ctx.GetPlace());
    auto *d_src_mask_out_data = d_src_mask_out->mutable_data<T>(ctx.GetPlace());

    // parameter grad
    auto *d_nonbatched_bias =
        ctx.Output<Tensor>(framework::GradVarName("NonbatchedBias"));

    if (nonbatched_bias != nullptr) {
      d_nonbatched_bias->mutable_data<T>(ctx.GetPlace());
    }

    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = ctx.Input<Tensor>("QKVWeight")->dims();

    int batch_size = input_x_dims[0];
    int seq_len_m = input_x_dims[1];
    int seq_len_r = input_x_dims[2];
    int hidden_size = input_x_dims[3];

    // qkv_weight[3, n_head, c, qkv_dim]
    int num_head = qkv_w_dims[1];
    int c = qkv_w_dims[2];

    // Gradient of Output Linear
    int m = batch_size * seq_len_m * seq_len_r;
    int n = hidden_size;
    int k = num_head * c;
    Tensor *d_out_linear_input =
        ComputeOutputLinearBackward<T>(ctx, m, n, k, is_gating);

    // Gradient of Gating Linear
    Tensor *d_fmha_out = nullptr;
    if (is_gating) {
      m = batch_size * seq_len_m * seq_len_r;
      n = num_head * c;
      k = hidden_size;
      // d_out_linear_input is d_gate_out.
      d_fmha_out =
          ComputeGatingLinearBackward<T>(ctx, d_out_linear_input, m, n, k);
    } else {
      d_fmha_out = d_out_linear_input;
    }

    auto fmha_ref_compute =
        FMHAGateRef<T>(ctx.cuda_device_context(), batch_size, seq_len_m,
                       seq_len_r, num_head, c);

    fmha_ref_compute.ComputeBackward(
        *transpose_out_2, src_mask, *softmax_out, *qk_out, *src_mask_out,
        *d_fmha_out, nonbatched_bias, d_nonbatched_bias, d_qktv_out,
        d_softmax_out, d_src_mask_out, d_qk_out, d_transpose_out_2, nullptr,
        d_qkv_out);

    // Gradient of Merged QKV Matmul
    m = batch_size * seq_len_m * seq_len_r;
    n = 3 * num_head * c;
    k = hidden_size;

    Tensor d_residual;
    d_residual.Resize(input_x_dims);
    d_residual.mutable_data<T>(ctx.GetPlace());
    ComputeMergedQKVMatmulBackward<T>(ctx, input_x, d_qkv_out, &d_residual, m,
                                      n, k);

    // Gradient accumulation
    std::vector<const Tensor *> ins = {&d_residual, d_x};
    std::vector<Tensor *> outs = {d_x};
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   T>(
        ctx.cuda_device_context(), ins, &outs, -1, AddFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_gate_attention,
                        ops::FusedGateAttentionOpKernel<float>,
                        ops::FusedGateAttentionOpKernel<double>,
                        ops::FusedGateAttentionOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(fused_gate_attention_grad,
                        ops::FusedGateAttentionGradKernel<float>,
                        ops::FusedGateAttentionGradKernel<double>,
                        ops::FusedGateAttentionGradKernel<plat::float16>);
