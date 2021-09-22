/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cuda_device_function.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/miopen_helper.h"
#endif

#include <cuda_fp16.h>
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/fused/fused_attention_op.h"

#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class FusedAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto *input_x = ctx.Input<Tensor>("X");

    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    auto *ln_scale = ctx.Input<Tensor>("LnScale");
    auto *ln_bias = ctx.Input<Tensor>("LnBias");
    auto *ln_mean = ctx.Output<Tensor>("LnMean");
    auto *ln_var = ctx.Output<Tensor>("LnVariance");
    auto *ln_out = ctx.Output<Tensor>("LnOut");

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto *qkv_weight = ctx.Input<Tensor>("QKVW");
    auto *qkv_bias = ctx.Input<Tensor>("QKVBias");
    auto *qkv_out = ctx.Output<Tensor>("QKVOut");
    auto *qkv_bias_out = ctx.Output<Tensor>("QKVBiasOut");

    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *transpose_out_2 = ctx.Output<Tensor>("TransposeOut2");
    auto *qk_out = ctx.Output<Tensor>("QKOut");
    auto *qktv_out = ctx.Output<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *attn_dropout_mask_out = ctx.Output<Tensor>("AttnDropoutMaskOut");
    auto *attn_dropout_out = ctx.Output<Tensor>("AttnDropoutOut");
    auto *src_mask_out = ctx.Output<Tensor>("SrcMaskOut");
    auto *fmha_out = ctx.Output<Tensor>("FMHAOut");

    auto *out_linear_weight = ctx.Input<Tensor>("OutLinearW");
    auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");
    auto *out_linear_out = ctx.Output<Tensor>("OutLinearOut");

    auto *ln_scale_2 = ctx.Input<Tensor>("Ln2Scale");
    auto *ln_bias_2 = ctx.Input<Tensor>("Ln2Bias");
    auto *dropout_mask_out = ctx.Output<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Output<Tensor>("BiasDropoutResidualOut");
    auto *ln_mean_2 = ctx.Output<Tensor>("Ln2Mean");
    auto *ln_var_2 = ctx.Output<Tensor>("Ln2Variance");
    const float ln2epsilon = ctx.Attr<float>("ln2epsilon");

    float attn_dropout_prob = ctx.Attr<float>("attn_dropout_prob");
    bool is_test_1 = ctx.Attr<bool>("is_test1");
    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("dropout_implementation1");
    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 = ctx.HasInput("Seed1") ? ctx.Input<Tensor>("Seed1") : nullptr;
    bool is_fix_seed_1 = ctx.Attr<bool>("fix_seed1");
    int seed_val_1 = ctx.Attr<int>("seed1");

    // final output.
    auto *out = ctx.Output<Tensor>("Y");

    // get data ptr for qkv part.
    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    auto *x_data = input_x->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
    auto *ln_mean_data = ln_mean->mutable_data<U>(ctx.GetPlace());
    auto *ln_var_data = ln_var->mutable_data<U>(ctx.GetPlace());
    auto *ln_out_data = ln_out->mutable_data<T>(ctx.GetPlace());

    auto *qkv_weight_data = qkv_weight->data<T>();
    auto *qkv_bias_data = qkv_bias->data<T>();
    auto *qkv_out_data = qkv_out->mutable_data<T>(ctx.GetPlace());
    auto *qkv_bias_out_data = qkv_bias_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for FMHA.
    auto *src_mask_data = (src_mask == nullptr ? nullptr : src_mask->data<T>());
    auto *transpose_out_2_data =
        transpose_out_2->mutable_data<T>(ctx.GetPlace());
    auto *qk_out_data = qk_out->mutable_data<T>(ctx.GetPlace());
    auto *qktv_out_data = qktv_out->mutable_data<T>(ctx.GetPlace());
    auto *src_mask_out_data = src_mask_out->mutable_data<T>(ctx.GetPlace());
    auto *softmax_out_data = softmax_out->mutable_data<T>(ctx.GetPlace());
    auto *attn_dropout_mask_out_data =
        attn_dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    auto *attn_dropout_out_data =
        attn_dropout_out->mutable_data<T>(ctx.GetPlace());
    auto *fmha_out_data = fmha_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for out_linear.
    auto *out_linear_weight_data = out_linear_weight->data<T>();
    auto *out_linear_bias_data = out_linear_bias->data<T>();
    auto *out_linear_out_data = out_linear_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for bias+dropout+residual+layernorm
    auto *ln_scale_2_data =
        (ln_scale_2 == nullptr ? nullptr : ln_scale_2->data<U>());
    auto *ln_bias_2_data =
        (ln_bias_2 == nullptr ? nullptr : ln_bias_2->data<U>());
    auto *dropout_mask_out_data =
        dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    auto *bias_dropout_residual_out_data =
        bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
    auto *ln_mean_2_data = ln_mean_2->mutable_data<U>(ctx.GetPlace());
    auto *ln_var_2_data = ln_var_2->mutable_data<U>(ctx.GetPlace());
    auto *final_out_data = out->mutable_data<T>(ctx.GetPlace());

    int batch_size = input_x_dims[0];
    int max_seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];

    int num_head = qkv_w_dims[1];
    int dim_head = qkv_w_dims[2];

    int bsz_seq = batch_size * max_seq_len;
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    bool transA = false;
    bool transB = true;
    bool compute_bias = true;
    auto layer_norm_compute = AttnLayerNorm<T>(ctx.cuda_device_context(),
                                               epsilon, bsz_seq, dim_embed);
    auto qkv_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), transA, transB, bsz_seq,
                      output_size, input_size, compute_bias);

    AttnDropoutParam attn_dropout_param(
        is_test_1, dropout_implementation_1, attn_dropout_prob,
        is_upscale_in_train_1, is_fix_seed_1, seed_val_1, seed_1);
    auto fmha_ref_compute =
        FMHARef<T>(ctx.cuda_device_context(), batch_size, max_seq_len, num_head,
                   dim_head, attn_dropout_param);

    output_size = hidden_size;
    transA = false;
    transB = false;
    compute_bias = false;
    auto out_linear_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), transA, transB, bsz_seq,
                      output_size, input_size, compute_bias);
    DropoutParam dropout_param2(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(), bsz_seq, dim_embed, dropout_param2,
        ln2epsilon);

    if (pre_layer_norm) {
      layer_norm_compute.ComputeForward(x_data, ln_scale_data, ln_bias_data,
                                        ln_out_data, ln_mean_data, ln_var_data);
      qkv_compute.ComputeForward(qkv_weight_data, ln_out_data, qkv_bias_data,
                                 qkv_out_data, qkv_bias_out_data);
    } else {
      qkv_compute.ComputeForward(qkv_weight_data, x_data, qkv_bias_data,
                                 qkv_out_data, qkv_bias_out_data);
    }
    fmha_ref_compute.ComputeForward(*qkv_bias_out, *src_mask, transpose_out_2,
                                    qk_out, src_mask_out, softmax_out,
                                    attn_dropout_mask_out, attn_dropout_out,
                                    qktv_out, fmha_out);
    // fmha_out: [batch_size, seq_len, num_head, head_dim]
    // weight: [1024, 1024], [embed_dim, embed_dim]
    // out_linear_out: [batch_size, seq_len, embed_dim]
    out_linear_compute.ComputeForward(out_linear_weight_data, fmha_out_data,
                                      nullptr, out_linear_out_data, nullptr);
    // out = layernorm(residual + dropout(src + bias))
    fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
        ctx.cuda_device_context(), out_linear_out_data, x_data,
        out_linear_bias_data, ln_scale_2_data, ln_bias_2_data,
        bias_dropout_residual_out_data, dropout_mask_out_data, final_out_data,
        ln_mean_2_data, ln_var_2_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_attention, ops::FusedAttentionOpKernel<float>,
                        ops::FusedAttentionOpKernel<double>,
                        ops::FusedAttentionOpKernel<plat::float16>);
