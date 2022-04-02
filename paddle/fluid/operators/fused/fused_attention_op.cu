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
static void AllReduce(framework::Tensor &tensor,  // NOLINT
                      const int ring_id,
                      const platform::CUDADeviceContext &ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(tensor.dtype()));
  int64_t numel = tensor.numel();
  const void *sendbuff = tensor.data<T>();
  auto place = ctx.GetPlace();
  void *recvbuff = tensor.mutable_data<T>(place);
  auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
  auto stream = ctx.stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

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
    auto *cache_kv = ctx.Input<Tensor>("CacheKV");
    auto *cache_kv_out = ctx.Output<Tensor>("CacheKVOut");
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
    const float ln_epsilon = ctx.Attr<float>("ln_epsilon");

    float attn_dropout_rate = ctx.Attr<float>("attn_dropout_rate");
    bool is_test_1 = ctx.Attr<bool>("attn_dropout_is_test");
    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("attn_dropout_implementation");
    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 = ctx.HasInput("Seed1") ? ctx.Input<Tensor>("Seed1") : nullptr;
    bool is_fix_seed_1 = ctx.Attr<bool>("attn_dropout_fix_seed");
    int seed_val_1 = ctx.Attr<int>("attn_dropout_seed");
    int ring_id = ctx.Attr<int>("ring_id");

    // final output.
    auto *out = ctx.Output<Tensor>("Y");

    // get data ptr for qkv part.
    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    auto *x_data = input_x->data<T>();
    auto *qkv_weight_data = qkv_weight->data<T>();
    auto *qkv_bias_data = (qkv_bias == nullptr) ? nullptr : qkv_bias->data<T>();
    auto *qkv_out_data = qkv_out->mutable_data<T>(ctx.GetPlace());
    auto *qkv_bias_out_data =
        (qkv_bias == nullptr) ? nullptr
                              : qkv_bias_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for FMHA.
    auto *transpose_out_2_data =
        transpose_out_2->mutable_data<T>(ctx.GetPlace());
    auto *cache_kv_out_data =
        (cache_kv_out == nullptr)
            ? nullptr
            : cache_kv_out->mutable_data<T>(ctx.GetPlace());
    auto *qk_out_data = qk_out->mutable_data<T>(ctx.GetPlace());
    auto *qktv_out_data = qktv_out->mutable_data<T>(ctx.GetPlace());
    auto *src_mask_out_data =
        (src_mask == nullptr) ? nullptr
                              : src_mask_out->mutable_data<T>(ctx.GetPlace());
    auto *softmax_out_data = softmax_out->mutable_data<T>(ctx.GetPlace());
    auto *attn_dropout_mask_out_data =
        attn_dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    auto *attn_dropout_out_data =
        attn_dropout_out->mutable_data<T>(ctx.GetPlace());
    auto *fmha_out_data = fmha_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for out_linear.
    auto *out_linear_weight_data = out_linear_weight->data<T>();
    auto *out_linear_bias_data =
        (out_linear_bias == nullptr) ? nullptr : out_linear_bias->data<T>();
    auto *out_linear_out_data = out_linear_out->mutable_data<T>(ctx.GetPlace());

    // get data ptr for bias+dropout+residual+layernorm
    auto *dropout_mask_out_data =
        dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
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

    bool compute_bias = true;
    if (qkv_bias == nullptr) {
      compute_bias = false;
    }
    // (transA, transB, compute_bias) = (false, true, true)
    auto qkv_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, true, bsz_seq,
                      output_size, input_size, compute_bias);

    AttnDropoutParam attn_dropout_param(
        is_test_1, dropout_implementation_1, attn_dropout_rate,
        is_upscale_in_train_1, is_fix_seed_1, seed_val_1, seed_1);
    auto fmha_ref_compute =
        FMHARef<T>(ctx.cuda_device_context(), batch_size, max_seq_len, num_head,
                   dim_head, attn_dropout_param);

    output_size = hidden_size;
    // (transA, transB, compute_bias) = (false, false, false)
    // NOTE(Yuang Liu): For general input size == output size, change the
    // position won't have effects. For mp, the output size is mp_head * dkey
    // which is actually the input size. While the input size is hidden size,
    // which is actually the output size. So for out linear, switch the
    // input size and output size.
    auto out_linear_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), false, false, bsz_seq,
                      input_size, output_size, false);
    DropoutParam dropout_param2(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(), bsz_seq, dim_embed, dropout_param2,
        ln_epsilon);

    if (pre_layer_norm) {
      auto *ln_scale_data =
          (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
      auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
      auto *ln_mean_data = ln_mean->mutable_data<U>(ctx.GetPlace());
      auto *ln_var_data = ln_var->mutable_data<U>(ctx.GetPlace());
      auto *ln_out_data = ln_out->mutable_data<T>(ctx.GetPlace());

      layer_norm_compute.ComputeForward(x_data, ln_scale_data, ln_bias_data,
                                        ln_out_data, ln_mean_data, ln_var_data);
      qkv_compute.ComputeForward(qkv_weight, ln_out, qkv_bias, qkv_out,
                                 qkv_bias_out);
    } else {
      qkv_compute.ComputeForward(qkv_weight, input_x, qkv_bias, qkv_out,
                                 qkv_bias_out);
    }
    if (qkv_bias == nullptr) {
      fmha_ref_compute.ComputeForward(
          *qkv_out, cache_kv, src_mask, transpose_out_2, cache_kv_out, qk_out,
          src_mask_out, softmax_out, attn_dropout_mask_out, attn_dropout_out,
          qktv_out, fmha_out);
    } else {
      fmha_ref_compute.ComputeForward(
          *qkv_bias_out, cache_kv, src_mask, transpose_out_2, cache_kv_out,
          qk_out, src_mask_out, softmax_out, attn_dropout_mask_out,
          attn_dropout_out, qktv_out, fmha_out);
    }

    // fmha_out: [batch_size, seq_len, num_head, head_dim]
    // weight:   [embed_dim, embed_dim]
    // out_linear_out: [batch_size, seq_len, embed_dim]
    out_linear_compute.ComputeForward(out_linear_weight, fmha_out, nullptr,
                                      out_linear_out, nullptr);
    // tensor model parallel
    AllReduce<T>(*out_linear_out, ring_id, ctx.cuda_device_context());

    if (pre_layer_norm) {
      // output = (residual + dropout(input + bias))
      fused_dropout_layernorm_helper.ResidualDropoutBias(
          ctx.cuda_device_context(), out_linear_out_data, x_data,
          out_linear_bias_data, final_out_data, dropout_mask_out_data);
    } else {
      auto *ln_scale_2_data =
          (ln_scale_2 == nullptr ? nullptr : ln_scale_2->data<U>());
      auto *ln_bias_2_data =
          (ln_bias_2 == nullptr ? nullptr : ln_bias_2->data<U>());
      auto *bias_dropout_residual_out_data =
          bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
      auto *ln_mean_2_data = ln_mean_2->mutable_data<U>(ctx.GetPlace());
      auto *ln_var_2_data = ln_var_2->mutable_data<U>(ctx.GetPlace());
      // output = layernorm(residual + dropout(input + bias))
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          ctx.cuda_device_context(), out_linear_out_data, x_data,
          out_linear_bias_data, ln_scale_2_data, ln_bias_2_data,
          bias_dropout_residual_out_data, dropout_mask_out_data, final_out_data,
          ln_mean_2_data, ln_var_2_data);
    }
  }
};

template <typename T>
class FusedAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    const float ln2epsilon = ctx.Attr<float>("ln_epsilon");

    float attn_dropout_prob = ctx.Attr<float>("attn_dropout_rate");
    bool is_test_1 = ctx.Attr<bool>("attn_dropout_is_test");
    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("attn_dropout_implementation");
    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 = ctx.HasInput("Seed1") ? ctx.Input<Tensor>("Seed1") : nullptr;
    bool is_fix_seed_1 = ctx.Attr<bool>("attn_dropout_fix_seed");
    int seed_val_1 = ctx.Attr<int>("attn_dropout_seed");
    int ring_id = ctx.Attr<int>("ring_id");

    // get inputs.
    auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *d_y_data = d_y->data<T>();

    // fw input
    auto *input_x = ctx.Input<Tensor>("X");
    auto *ln_scale = ctx.Input<Tensor>("LnScale");
    auto *ln_2_scale = ctx.Input<Tensor>("Ln2Scale");
    auto *x_data = input_x->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_2_scale_data =
        (ln_2_scale == nullptr ? nullptr : ln_2_scale->data<U>());
    // fw parameters.
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto *qkv_weight = ctx.Input<Tensor>("QKVW");
    auto *qkv_bias = ctx.Input<Tensor>("QKVBias");
    auto *out_linear_weight = ctx.Input<Tensor>("OutLinearW");
    auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");
    auto *src_mask_data = (src_mask == nullptr ? nullptr : src_mask->data<T>());
    auto *qkv_weight_data = qkv_weight->data<T>();
    auto *qkv_bias_data = (qkv_bias == nullptr) ? nullptr : qkv_bias->data<T>();
    auto *out_linear_weight_data = out_linear_weight->data<T>();
    auto *out_linear_bias_data =
        (out_linear_bias == nullptr) ? nullptr : out_linear_bias->data<T>();

    // fw output
    auto *fmha_out = ctx.Input<Tensor>("FMHAOut");
    auto *transpose_out_2 = ctx.Input<Tensor>("TransposeOut2");
    auto *qk_out = ctx.Input<Tensor>("QKOut");
    auto *qktv_out = ctx.Input<Tensor>("QKTVOut");
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");
    auto *attn_dropout_mask_out = ctx.Input<Tensor>("AttnDropoutMaskOut");
    auto *attn_dropout_out = ctx.Input<Tensor>("AttnDropoutOut");
    auto *src_mask_out = ctx.Input<Tensor>("SrcMaskOut");
    auto *out_linear_out = ctx.Input<Tensor>("OutLinearOut");
    auto *ln_2_mean = ctx.Input<Tensor>("Ln2Mean");
    auto *ln_2_var = ctx.Input<Tensor>("Ln2Variance");
    auto *dropout_mask_out = ctx.Input<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Input<Tensor>("BiasDropoutResidualOut");
    auto *fmha_out_data = fmha_out->data<T>();
    auto *transpose_out_2_data = transpose_out_2->data<T>();
    auto *qk_out_data = qk_out->data<T>();
    auto *qktv_out_data = qktv_out->data<T>();
    auto *softmax_out_data = softmax_out->data<T>();
    auto *src_mask_out_data =
        (src_mask == nullptr) ? nullptr : src_mask_out->data<T>();
    auto *out_linear_out_data = out_linear_out->data<T>();
    auto *dropout_mask_out_data = dropout_mask_out->data<uint8_t>();

    // output's grad
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_qkv_out = ctx.Output<Tensor>(framework::GradVarName("QKVOut"));
    auto *d_qkv_bias_out =
        ctx.Output<Tensor>(framework::GradVarName("QKVBiasOut"));
    auto *d_qktv_out = ctx.Output<Tensor>(framework::GradVarName("QKTVOut"));
    auto *d_transpose_out_2 =
        ctx.Output<Tensor>(framework::GradVarName("TransposeOut2"));
    auto *d_qk_out = ctx.Output<Tensor>(framework::GradVarName("QKOut"));
    auto *d_softmax_out =
        ctx.Output<Tensor>(framework::GradVarName("SoftmaxOut"));
    auto *d_attn_dropout_out =
        ctx.Output<Tensor>(framework::GradVarName("AttnDropoutOut"));
    auto *d_src_mask_out =
        ctx.Output<Tensor>(framework::GradVarName("SrcMaskOut"));
    auto *d_fmha_out = ctx.Output<Tensor>(framework::GradVarName("FMHAOut"));
    auto *d_out_linear_out =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearOut"));
    auto *d_bias_dropout_residual_out =
        ctx.Output<Tensor>(framework::GradVarName("BiasDropoutResidualOut"));
    auto *d_x_data = d_x->mutable_data<T>(ctx.GetPlace());
    // when qkv_bias is not nullptr, d_qkv_out is equals to d_qkv_bias_out, the
    // space can be reused.
    auto *d_qkv_out_data = (d_qkv_bias_out != nullptr)
                               ? nullptr
                               : d_qkv_out->mutable_data<T>(ctx.GetPlace());
    auto *d_qkv_bias_out_data =
        (d_qkv_bias_out == nullptr)
            ? nullptr
            : d_qkv_bias_out->mutable_data<T>(ctx.GetPlace());
    auto *d_qktv_out_data = d_qktv_out->mutable_data<T>(ctx.GetPlace());
    auto *d_transpose_out_2_data =
        d_transpose_out_2->mutable_data<T>(ctx.GetPlace());
    auto *d_qk_out_data = d_qk_out->mutable_data<T>(ctx.GetPlace());
    auto *d_softmax_out_data = d_softmax_out->mutable_data<T>(ctx.GetPlace());
    auto *d_attn_dropout_out_data =
        d_attn_dropout_out->mutable_data<T>(ctx.GetPlace());
    auto *d_src_mask_out_data =
        (src_mask == nullptr) ? nullptr
                              : d_src_mask_out->mutable_data<T>(ctx.GetPlace());
    auto *d_fmha_out_data = d_fmha_out->mutable_data<T>(ctx.GetPlace());
    auto *d_out_linear_out_data =
        d_out_linear_out->mutable_data<T>(ctx.GetPlace());

    // parameter grad
    auto *d_qkv_weight = ctx.Output<Tensor>(framework::GradVarName("QKVW"));
    auto *d_qkv_bias = ctx.Output<Tensor>(framework::GradVarName("QKVBias"));
    auto *d_out_linear_weight =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearW"));
    auto *d_out_linear_bias =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));
    auto *d_ln_2_scale = ctx.Output<Tensor>(framework::GradVarName("Ln2Scale"));
    auto *d_ln_2_bias = ctx.Output<Tensor>(framework::GradVarName("Ln2Bias"));

    auto *d_qkv_weight_data = d_qkv_weight->mutable_data<T>(ctx.GetPlace());
    auto *d_qkv_bias_data = (d_qkv_bias == nullptr)
                                ? nullptr
                                : d_qkv_bias->mutable_data<T>(ctx.GetPlace());
    auto *d_out_linear_weight_data =
        d_out_linear_weight->mutable_data<T>(ctx.GetPlace());
    auto *d_out_linear_bias_data =
        (d_out_linear_bias == nullptr)
            ? nullptr
            : d_out_linear_bias->mutable_data<T>(ctx.GetPlace());

    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    int batch_size = input_x_dims[0];
    int max_seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int num_head = qkv_w_dims[1];
    int dim_head = qkv_w_dims[2];

    int bsz_seq = batch_size * max_seq_len;
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    Tensor d_residual;
    d_residual.Resize(input_x_dims);
    T *d_residual_data = d_residual.mutable_data<T>(ctx.GetPlace());

    bool transA = false;
    bool transB = true;
    bool compute_qkv_bias = true;
    if (qkv_bias == nullptr) {
      compute_qkv_bias = false;
    }
    auto layer_norm_compute = AttnLayerNorm<T>(ctx.cuda_device_context(),
                                               epsilon, bsz_seq, dim_embed);
    auto qkv_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), transA, transB, bsz_seq,
                      output_size, input_size, compute_qkv_bias);
    AttnDropoutParam attn_dropout_param(
        is_test_1, dropout_implementation_1, attn_dropout_prob,
        is_upscale_in_train_1, is_fix_seed_1, seed_val_1, seed_1);
    auto fmha_ref_compute =
        FMHARef<T>(ctx.cuda_device_context(), batch_size, max_seq_len, num_head,
                   dim_head, attn_dropout_param);
    output_size = hidden_size;
    transA = false;
    transB = false;
    bool compute_bias = false;
    // (b*s, num_head * dim_head) * (num_head * dim_head, dim_embed)
    auto out_linear_compute =
        AttnMatMul<T>(ctx.cuda_device_context(), transA, transB, bsz_seq,
                      input_size, output_size, compute_bias);
    DropoutParam dropout_param2(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(), bsz_seq, dim_embed, dropout_param2,
        ln2epsilon);

    if (pre_layer_norm) {
      fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
          ctx.cuda_device_context(), d_y_data, dropout_mask_out_data,
          d_out_linear_out_data, d_residual_data, d_out_linear_bias_data);
    } else {
      auto *ln_2_mean_data = ln_2_mean->data<U>();
      auto *ln_2_var_data = ln_2_var->data<U>();
      auto *bias_dropout_residual_out_data =
          bias_dropout_residual_out->data<T>();
      auto *d_ln_2_scale_data =
          (d_ln_2_scale == nullptr ? nullptr : d_ln_2_scale->mutable_data<U>(
                                                   ctx.GetPlace()));
      auto *d_ln_2_bias_data =
          (d_ln_2_bias == nullptr ? nullptr : d_ln_2_bias->mutable_data<U>(
                                                  ctx.GetPlace()));
      auto *d_bias_dropout_residual_out_data =
          d_bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());

      fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
          ctx.cuda_device_context(), d_y_data, bias_dropout_residual_out_data,
          dropout_mask_out_data, ln_2_scale_data, ln_2_mean_data, ln_2_var_data,
          d_bias_dropout_residual_out_data, d_ln_2_scale_data, d_ln_2_bias_data,
          d_out_linear_out_data, d_out_linear_bias_data, d_residual_data);
    }

    out_linear_compute.ComputeBackward(fmha_out, out_linear_weight,
                                       d_out_linear_out, d_fmha_out,
                                       d_out_linear_weight, nullptr);

    if (qkv_bias != nullptr) {
      fmha_ref_compute.ComputeBackward(
          *transpose_out_2, src_mask, *softmax_out, *attn_dropout_mask_out,
          *attn_dropout_out, *qk_out, *src_mask_out, *d_fmha_out, d_qktv_out,
          d_attn_dropout_out, d_softmax_out, d_src_mask_out, d_qk_out,
          d_transpose_out_2, nullptr, d_qkv_bias_out);
    } else {
      fmha_ref_compute.ComputeBackward(
          *transpose_out_2, src_mask, *softmax_out, *attn_dropout_mask_out,
          *attn_dropout_out, *qk_out, *src_mask_out, *d_fmha_out, d_qktv_out,
          d_attn_dropout_out, d_softmax_out, d_src_mask_out, d_qk_out,
          d_transpose_out_2, nullptr, d_qkv_out);
    }

    if (pre_layer_norm) {
      auto *ln_mean = ctx.Input<Tensor>("LnMean");
      auto *ln_var = ctx.Input<Tensor>("LnVariance");
      auto *ln_out = ctx.Input<Tensor>("LnOut");
      auto *ln_mean_data = ln_mean->data<U>();
      auto *ln_var_data = ln_var->data<U>();
      auto *ln_out_data = ln_out->data<T>();

      auto *d_ln_out = ctx.Output<Tensor>(framework::GradVarName("LnOut"));
      auto *d_ln_scale = ctx.Output<Tensor>(framework::GradVarName("LnScale"));
      auto *d_ln_bias = ctx.Output<Tensor>(framework::GradVarName("LnBias"));
      auto *d_ln_out_data = d_ln_out->mutable_data<T>(ctx.GetPlace());
      auto *d_ln_scale_data =
          (d_ln_scale == nullptr ? nullptr
                                 : d_ln_scale->mutable_data<U>(ctx.GetPlace()));
      auto *d_ln_bias_data =
          (d_ln_bias == nullptr ? nullptr
                                : d_ln_bias->mutable_data<U>(ctx.GetPlace()));
      if (qkv_bias != nullptr) {
        qkv_compute.ComputeBackward(ln_out, qkv_weight, d_qkv_bias_out,
                                    d_ln_out, d_qkv_weight, d_qkv_bias);
      } else {
        qkv_compute.ComputeBackward(ln_out, qkv_weight, d_qkv_out, d_ln_out,
                                    d_qkv_weight, d_qkv_bias);
      }
      // tensor model parallel
      AllReduce<T>(*d_ln_out, ring_id, ctx.cuda_device_context());
      layer_norm_compute.ComputeBackward(x_data, d_ln_out_data, ln_scale_data,
                                         ln_mean_data, ln_var_data, d_x_data,
                                         d_ln_scale_data, d_ln_bias_data);
    } else {
      if (qkv_bias != nullptr) {
        qkv_compute.ComputeBackward(input_x, qkv_weight, d_qkv_bias_out, d_x,
                                    d_qkv_weight, d_qkv_bias);
      } else {
        qkv_compute.ComputeBackward(input_x, qkv_weight, d_qkv_out, d_x,
                                    d_qkv_weight, d_qkv_bias);
      }
      // tensor model parallel
      AllReduce<T>(*d_x, ring_id, ctx.cuda_device_context());
    }
    // gradient accumulation
    std::vector<const Tensor *> ins;
    std::vector<Tensor *> outs;
    ins.emplace_back(&d_residual);
    ins.emplace_back(d_x);
    outs.emplace_back(d_x);
    int elewise_add_axis = -1;
    paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T,
                                                   T>(
        ctx.cuda_device_context(), ins, &outs, elewise_add_axis,
        AddFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_attention, ops::FusedAttentionOpKernel<float>,
                        ops::FusedAttentionOpKernel<double>,
                        ops::FusedAttentionOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(fused_attention_grad,
                        ops::FusedAttentionGradKernel<float>,
                        ops::FusedAttentionGradKernel<double>,
                        ops::FusedAttentionGradKernel<plat::float16>);
