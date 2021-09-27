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

#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cudnn_helper.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/math/math_function.h"

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
    auto *pre_ln_scale = ctx.Input<Tensor>("PreLnScale");
    auto *pre_ln_bias = ctx.Input<Tensor>("PreLnBias");
    auto *pre_ln_mean = ctx.Output<Tensor>("PreLnMean");
    auto *pre_ln_var = ctx.Output<Tensor>("PreLnVariance");
    auto *pre_ln_out = ctx.Output<Tensor>("PreLnOut");

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto *qkv_weight = ctx.Input<Tensor>("QKVW");
    auto *qkv_bias = ctx.Input<Tensor>("QKVBias");
    auto *qkv_out = ctx.Output<Tensor>("QKVOut");
    auto *qkv_bias_out = ctx.Output<Tensor>("QKVBiasOut");

    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *transpose_out = ctx.Output<Tensor>("TransposeOut");
    auto *qk_out = ctx.Output<Tensor>("QKOut");
    auto *qktv_out = ctx.Output<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *attn_dropout_mask_out = ctx.Output<Tensor>("AttnDropoutMaskOut");
    auto *attn_dropout_out = ctx.Output<Tensor>("AttnDropoutOut");
    auto *src_mask_out = ctx.Output<Tensor>("SrcMaskOut");
    auto *fmha_out = ctx.Output<Tensor>("FMHAOut");

    auto *linear_weight = ctx.Input<Tensor>("LinearW");
    auto *linear_bias = ctx.Input<Tensor>("LinearBias");
    auto *linear_out = ctx.Output<Tensor>("LinearOut");

    auto *ln_scale = ctx.Input<Tensor>("LnScale");
    auto *ln_bias = ctx.Input<Tensor>("LnBias");
    auto *dropout_mask_out = ctx.Output<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Output<Tensor>("BiasDropoutResidualOut");
    auto *ln_mean = ctx.Output<Tensor>("LnMean");
    auto *ln_var = ctx.Output<Tensor>("LnVariance");
    const float ln_epsilon = ctx.Attr<float>("ln_epsilon");

    float attn_dropout_prob = ctx.Attr<float>("attn_dropout_prob");
    bool attn_dropout_is_test = ctx.Attr<bool>("attn_dropout_is_test");
    auto &attn_dropout_implementation =
        ctx.Attr<std::string>("attn_dropout_implementation");
    bool attn_dropout_is_upscale_in_train =
        (attn_dropout_implementation == "upscale_in_train");
    auto *attn_dropout_seed = ctx.HasInput("AttnDropoutSeed")
                                  ? ctx.Input<Tensor>("AttnDropoutSeed")
                                  : nullptr;
    bool attn_dropout_fix_seed = ctx.Attr<bool>("attn_dropout_fix_seed");
    int attn_dropout_seed_val = ctx.Attr<int>("attn_dropout_seed_val");

    // final output.
    auto *out = ctx.Output<Tensor>("Y");

    // get data ptr for qkv part.
    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    auto *x_data = input_x->data<T>();
    auto *pre_ln_scale_data =
        (pre_ln_scale == nullptr ? nullptr : pre_ln_scale->data<U>());
    auto *pre_ln_bias_data =
        (pre_ln_bias == nullptr ? nullptr : pre_ln_bias->data<U>());
    auto *pre_ln_mean_data = pre_ln_mean->mutable_data<U>(ctx.GetPlace());
    auto *pre_ln_var_data = pre_ln_var->mutable_data<U>(ctx.GetPlace());
    auto *pre_ln_out_data = pre_ln_out->mutable_data<T>(ctx.GetPlace());

    auto *qkv_weight_data = qkv_weight->data<T>();
    auto *qkv_bias_data = qkv_bias->data<T>();
    auto *qkv_out_data = qkv_out->mutable_data<T>(ctx.GetPlace());
    auto *qkv_bias_out_data = qkv_bias_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for FMHA.
    auto *transpose_out_data = transpose_out->mutable_data<T>(ctx.GetPlace());
    auto *qk_out_data = qk_out->mutable_data<T>(ctx.GetPlace());
    auto *qktv_out_data = qktv_out->mutable_data<T>(ctx.GetPlace());
    auto *src_mask_out_data = src_mask_out->mutable_data<T>(ctx.GetPlace());
    auto *softmax_out_data = softmax_out->mutable_data<T>(ctx.GetPlace());
    auto *attn_dropout_mask_out_data =
        attn_dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    auto *attn_dropout_out_data =
        attn_dropout_out->mutable_data<T>(ctx.GetPlace());
    auto *fmha_out_data = fmha_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for linear.
    auto *linear_weight_data = linear_weight->data<T>();
    auto *linear_bias_data = linear_bias->data<T>();
    auto *linear_out_data = linear_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for bias+dropout+residual+layernorm
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
    auto *dropout_mask_out_data =
        dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    auto *bias_dropout_residual_out_data =
        bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
    auto *ln_mean_data = ln_mean->mutable_data<U>(ctx.GetPlace());
    auto *ln_var_data = ln_var->mutable_data<U>(ctx.GetPlace());
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

    auto layer_norm_compute = AttnLayerNorm<T>(ctx.cuda_device_context(),
                                               epsilon, bsz_seq, dim_embed);
    // (transA, transB, compute_bias) = (false, true, true)
    auto qkv_compute = AttnMatMul<T>(ctx.cuda_device_context(), false, true,
                                     bsz_seq, output_size, input_size, true);

    AttnDropoutParam attn_dropout_param(
        attn_dropout_is_test, attn_dropout_implementation, attn_dropout_prob,
        attn_dropout_is_upscale_in_train, attn_dropout_fix_seed,
        attn_dropout_seed_val, attn_dropout_seed);
    auto fmha_ref_compute =
        FMHARef<T>(ctx.cuda_device_context(), batch_size, max_seq_len, num_head,
                   dim_head, attn_dropout_param);

    output_size = hidden_size;
    // (transA, transB, compute_bias) = (false, false, false)
    auto linear_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, bsz_seq,
                      output_size, input_size, false);
    DropoutParam dropout_param(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(), bsz_seq, dim_embed, dropout_param,
        ln_epsilon);

    if (pre_layer_norm) {
      layer_norm_compute.ComputeForward(x_data, pre_ln_scale_data,
                                        pre_ln_bias_data, pre_ln_out_data,
                                        pre_ln_mean_data, pre_ln_var_data);
      qkv_compute.ComputeForward(qkv_weight_data, pre_ln_out_data,
                                 qkv_bias_data, qkv_out_data,
                                 qkv_bias_out_data);
    } else {
      qkv_compute.ComputeForward(qkv_weight_data, x_data, qkv_bias_data,
                                 qkv_out_data, qkv_bias_out_data);
    }
    fmha_ref_compute.ComputeForward(*qkv_bias_out, *src_mask, transpose_out,
                                    qk_out, src_mask_out, softmax_out,
                                    attn_dropout_mask_out, attn_dropout_out,
                                    qktv_out, fmha_out);
    // fmha_out: [batch_size, seq_len, num_head, head_dim]
    // weight:   [embed_dim, embed_dim]
    // linear_out: [batch_size, seq_len, embed_dim]
    linear_compute.ComputeForward(linear_weight_data, fmha_out_data, nullptr,
                                  linear_out_data, nullptr);
    // output = layernorm(residual + dropout(input + bias))
    fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
        ctx.cuda_device_context(), linear_out_data, x_data, linear_bias_data,
        ln_scale_data, ln_bias_data, bias_dropout_residual_out_data,
        dropout_mask_out_data, final_out_data, ln_mean_data, ln_var_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_attention, ops::FusedAttentionOpKernel<float>,
                        ops::FusedAttentionOpKernel<double>,
                        ops::FusedAttentionOpKernel<plat::float16>);
