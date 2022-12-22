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
#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group_nccl.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const phi::GPUContext &ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup *pg = map->get(ring_id);
    auto pg_nccl = static_cast<distributed::ProcessGroupNCCL *>(pg);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::SUM;
    auto task = pg_nccl->AllReduce(&tensor, tensor, opts, true, true);
    task->Wait();
  } else {
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    const void *sendbuff = tensor.data<T>();
    auto place = ctx.GetPlace();
    void *recvbuff = ctx.template Alloc<T>(&tensor, tensor.numel() * sizeof(T));
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto stream = ctx.stream();
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
  }
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
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    auto *ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
    auto *ln_bias = ctx.Input<phi::DenseTensor>("LnBias");
    auto *ln_mean = ctx.Output<phi::DenseTensor>("LnMean");
    auto *ln_var = ctx.Output<phi::DenseTensor>("LnVariance");
    auto *ln_out = ctx.Output<phi::DenseTensor>("LnOut");

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVW");
    auto *qkv_bias = ctx.Input<phi::DenseTensor>("QKVBias");
    auto *qkv_out = ctx.Output<phi::DenseTensor>("QKVOut");
    auto *qkv_bias_out = ctx.Output<phi::DenseTensor>("QKVBiasOut");

    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    auto *transpose_out_2 = ctx.Output<phi::DenseTensor>("TransposeOut2");
    auto *cache_kv = ctx.Input<phi::DenseTensor>("CacheKV");
    auto *cache_kv_out = ctx.Output<phi::DenseTensor>("CacheKVOut");
    auto *qk_out = ctx.Output<phi::DenseTensor>("QKOut");
    auto *qktv_out = ctx.Output<phi::DenseTensor>("QKTVOut");
    auto *softmax_out = ctx.Output<phi::DenseTensor>("SoftmaxOut");
    auto *attn_dropout_mask_out =
        ctx.Output<phi::DenseTensor>("AttnDropoutMaskOut");
    auto *attn_dropout_out = ctx.Output<phi::DenseTensor>("AttnDropoutOut");
    auto *src_mask_out = ctx.Output<phi::DenseTensor>("SrcMaskOut");
    auto *fmha_out = ctx.Output<phi::DenseTensor>("FMHAOut");

    auto *out_linear_weight = ctx.Input<phi::DenseTensor>("OutLinearW");
    auto *out_linear_bias = ctx.Input<phi::DenseTensor>("OutLinearBias");
    auto *out_linear_out = ctx.Output<phi::DenseTensor>("OutLinearOut");

    auto *ln_scale_2 = ctx.Input<phi::DenseTensor>("Ln2Scale");
    auto *ln_bias_2 = ctx.Input<phi::DenseTensor>("Ln2Bias");
    auto *dropout_mask_out = ctx.Output<phi::DenseTensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Output<phi::DenseTensor>("BiasDropoutResidualOut");
    auto *ln_mean_2 = ctx.Output<phi::DenseTensor>("Ln2Mean");
    auto *ln_var_2 = ctx.Output<phi::DenseTensor>("Ln2Variance");
    const float ln_epsilon = ctx.Attr<float>("ln_epsilon");

    float attn_dropout_rate = ctx.Attr<float>("attn_dropout_rate");
    const bool has_attn_dropout = (attn_dropout_rate != 0.0f);
    DropoutParam dropout_param2(ctx, 0);
    const bool has_dropout = (dropout_param2.dropout_prob != 0.0f);

    bool is_test_1 = ctx.Attr<bool>("is_test");
    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("attn_dropout_implementation");
    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 =
        ctx.HasInput("Seed1") ? ctx.Input<phi::DenseTensor>("Seed1") : nullptr;
    bool is_fix_seed_1 = ctx.Attr<bool>("attn_dropout_fix_seed");
    int seed_val_1 = ctx.Attr<int>("attn_dropout_seed");
    int ring_id = ctx.Attr<int>("ring_id");

    // final output.
    auto *out = ctx.Output<phi::DenseTensor>("Y");

    // get data ptr for qkv part.
    const auto input_x_dims = input_x->dims();
    const auto qkv_w_dims = qkv_weight->dims();

    auto *x_data = input_x->data<T>();
    auto *qkv_weight_data = qkv_weight->data<T>();
    auto *qkv_bias_data = (qkv_bias == nullptr) ? nullptr : qkv_bias->data<T>();
    auto *qkv_out_data =
        dev_ctx.template Alloc<T>(qkv_out, qkv_out->numel() * sizeof(T));
    auto *qkv_bias_out_data =
        (qkv_bias == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(qkv_bias_out,
                                        qkv_bias_out->numel() * sizeof(T));

    // get data ptr for FMHA.
    auto *transpose_out_2_data = dev_ctx.template Alloc<T>(
        transpose_out_2, transpose_out_2->numel() * sizeof(T));
    auto *cache_kv_out_data =
        (cache_kv_out == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(cache_kv_out,
                                        cache_kv_out->numel() * sizeof(T));
    auto *qk_out_data =
        dev_ctx.template Alloc<T>(qk_out, qk_out->numel() * sizeof(T));
    auto *qktv_out_data =
        dev_ctx.template Alloc<T>(qktv_out, qktv_out->numel() * sizeof(T));
    auto *src_mask_out_data =
        (src_mask == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(src_mask_out,
                                        src_mask_out->numel() * sizeof(T));
    auto *softmax_out_data = dev_ctx.template Alloc<T>(
        softmax_out, softmax_out->numel() * sizeof(T));
    auto *attn_dropout_mask_out_data =
        has_attn_dropout ? dev_ctx.template Alloc<uint8_t>(
                               attn_dropout_mask_out,
                               attn_dropout_mask_out->numel() * sizeof(uint8_t))
                         : nullptr;
    auto *attn_dropout_out_data =
        has_attn_dropout
            ? dev_ctx.template Alloc<T>(attn_dropout_out,
                                        attn_dropout_out->numel() * sizeof(T))
            : nullptr;
    auto *fmha_out_data =
        dev_ctx.template Alloc<T>(fmha_out, fmha_out->numel() * sizeof(T));

    // get data ptr for out_linear.
    auto *out_linear_weight_data = out_linear_weight->data<T>();
    auto *out_linear_bias_data =
        (out_linear_bias == nullptr) ? nullptr : out_linear_bias->data<T>();
    auto *out_linear_out_data = dev_ctx.template Alloc<T>(
        out_linear_out, out_linear_out->numel() * sizeof(T));

    // get data ptr for bias+dropout+residual+layernorm
    auto *dropout_mask_out_data =
        has_dropout
            ? dev_ctx.template Alloc<uint8_t>(
                  dropout_mask_out, dropout_mask_out->numel() * sizeof(uint8_t))
            : nullptr;
    auto *final_out_data =
        dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    int batch_size = input_x_dims[0];
    int max_seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];

    int num_head = qkv_w_dims[1];
    int dim_head = qkv_w_dims[2];

    int bsz_seq = batch_size * max_seq_len;
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    auto layer_norm_compute = AttnLayerNorm<T>(
        ctx.cuda_device_context(), epsilon, bsz_seq, dim_embed);

    bool compute_bias = true;
    if (qkv_bias == nullptr) {
      compute_bias = false;
    }
    // (transA, transB, compute_bias) = (false, true, true)
    auto qkv_compute = AttnMatMul<T>(ctx.cuda_device_context(),
                                     false,
                                     true,
                                     bsz_seq,
                                     output_size,
                                     input_size,
                                     compute_bias);

    AttnDropoutParam attn_dropout_param(is_test_1,
                                        dropout_implementation_1,
                                        attn_dropout_rate,
                                        is_upscale_in_train_1,
                                        is_fix_seed_1,
                                        seed_val_1,
                                        seed_1);
    auto fmha_ref_compute = FMHARef<T>(ctx.cuda_device_context(),
                                       batch_size,
                                       max_seq_len,
                                       num_head,
                                       dim_head,
                                       attn_dropout_param);

    output_size = hidden_size;
    // (transA, transB, compute_bias) = (false, false, false)
    // NOTE(Yuang Liu): For general input size == output size, change the
    // position won't have effects. For mp, the output size is mp_head * dkey
    // which is actually the input size. While the input size is hidden size,
    // which is actually the output size. So for out linear, switch the
    // input size and output size.
    auto out_linear_compute = AttnMatMul<T>(ctx.cuda_device_context(),
                                            false,
                                            false,
                                            bsz_seq,
                                            input_size,
                                            output_size,
                                            false);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(),
        bsz_seq,
        dim_embed,
        dropout_param2,
        ln_epsilon);

    if (pre_layer_norm) {
      auto *ln_scale_data =
          (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
      auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
      auto *ln_mean_data =
          dev_ctx.template Alloc<U>(ln_mean, ln_mean->numel() * sizeof(U));
      auto *ln_var_data =
          dev_ctx.template Alloc<U>(ln_var, ln_var->numel() * sizeof(U));
      auto *ln_out_data =
          dev_ctx.template Alloc<T>(ln_out, ln_out->numel() * sizeof(T));

      layer_norm_compute.ComputeForward(x_data,
                                        ln_scale_data,
                                        ln_bias_data,
                                        ln_out_data,
                                        ln_mean_data,
                                        ln_var_data);
      qkv_compute.ComputeForward(
          qkv_weight, ln_out, qkv_bias, qkv_out, qkv_bias_out);
    } else {
      qkv_compute.ComputeForward(
          qkv_weight, input_x, qkv_bias, qkv_out, qkv_bias_out);
    }
    if (qkv_bias == nullptr) {
      fmha_ref_compute.ComputeForward(*qkv_out,
                                      cache_kv,
                                      src_mask,
                                      transpose_out_2,
                                      cache_kv_out,
                                      qk_out,
                                      src_mask_out,
                                      softmax_out,
                                      attn_dropout_mask_out,
                                      attn_dropout_out,
                                      qktv_out,
                                      fmha_out);
    } else {
      fmha_ref_compute.ComputeForward(*qkv_bias_out,
                                      cache_kv,
                                      src_mask,
                                      transpose_out_2,
                                      cache_kv_out,
                                      qk_out,
                                      src_mask_out,
                                      softmax_out,
                                      attn_dropout_mask_out,
                                      attn_dropout_out,
                                      qktv_out,
                                      fmha_out);
    }

    // fmha_out: [batch_size, seq_len, num_head, head_dim]
    // weight:   [embed_dim, embed_dim]
    // out_linear_out: [batch_size, seq_len, embed_dim]
    out_linear_compute.ComputeForward(
        out_linear_weight, fmha_out, nullptr, out_linear_out, nullptr);
    // tensor model parallel
    AllReduce<T>(*out_linear_out, ring_id, ctx.cuda_device_context());

    bool add_residual = ctx.Attr<bool>("add_residual");
    const T *residual_ptr = add_residual ? x_data : nullptr;
    if (pre_layer_norm) {
      // output = (residual + dropout(input + bias))
      fused_dropout_layernorm_helper.ResidualDropoutBias(
          ctx.cuda_device_context(),
          out_linear_out_data,
          residual_ptr,
          out_linear_bias_data,
          final_out_data,
          dropout_mask_out_data);
    } else {
      // TODO(Xreki): support post layer_norm case when add_residual is false.
      PADDLE_ENFORCE_EQ(add_residual,
                        true,
                        platform::errors::InvalidArgument(
                            "Attribute add_residual is expected to be true "
                            "when pre_layer_norm is false."));

      const U *ln_scale_2_ptr = ln_scale_2 ? ln_scale_2->data<U>() : nullptr;
      const U *ln_bias_2_ptr = ln_bias_2 ? ln_bias_2->data<U>() : nullptr;
      T *bias_dropout_residual_out_ptr = dev_ctx.template Alloc<T>(
          bias_dropout_residual_out,
          bias_dropout_residual_out->numel() * sizeof(T));
      U *ln_mean_2_ptr =
          dev_ctx.template Alloc<U>(ln_mean_2, ln_mean_2->numel() * sizeof(U));
      U *ln_var_2_ptr =
          dev_ctx.template Alloc<U>(ln_var_2, ln_var_2->numel() * sizeof(U));
      // output = layernorm(residual + dropout(input + bias))
      fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
          ctx.cuda_device_context(),
          out_linear_out_data,
          residual_ptr,
          out_linear_bias_data,
          ln_scale_2_ptr,
          ln_bias_2_ptr,
          bias_dropout_residual_out_ptr,
          dropout_mask_out_data,
          final_out_data,
          ln_mean_2_ptr,
          ln_var_2_ptr);
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

    const float attn_dropout_prob = ctx.Attr<float>("attn_dropout_rate");
    const bool has_attn_dropout = (attn_dropout_prob != 0.0f);
    DropoutParam dropout_param2(ctx, 0);
    const bool has_dropout = (dropout_param2.dropout_prob != 0.0f);

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    bool is_test_1 = ctx.Attr<bool>("is_test");
    auto &dropout_implementation_1 =
        ctx.Attr<std::string>("attn_dropout_implementation");
    bool is_upscale_in_train_1 =
        (dropout_implementation_1 == "upscale_in_train");
    auto *seed_1 =
        ctx.HasInput("Seed1") ? ctx.Input<phi::DenseTensor>("Seed1") : nullptr;
    bool is_fix_seed_1 = ctx.Attr<bool>("attn_dropout_fix_seed");
    int seed_val_1 = ctx.Attr<int>("attn_dropout_seed");
    int ring_id = ctx.Attr<int>("ring_id");

    // get inputs.
    auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto *d_y_data = d_y->data<T>();

    // fw input
    auto *input_x = ctx.Input<phi::DenseTensor>("X");
    auto *ln_scale = ctx.Input<phi::DenseTensor>("LnScale");
    auto *ln_2_scale = ctx.Input<phi::DenseTensor>("Ln2Scale");
    auto *x_data = input_x->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_2_scale_data =
        (ln_2_scale == nullptr ? nullptr : ln_2_scale->data<U>());
    // fw parameters.
    auto *src_mask = ctx.Input<phi::DenseTensor>("SrcMask");
    auto *qkv_weight = ctx.Input<phi::DenseTensor>("QKVW");
    auto *qkv_bias = ctx.Input<phi::DenseTensor>("QKVBias");
    auto *out_linear_weight = ctx.Input<phi::DenseTensor>("OutLinearW");
    auto *out_linear_bias = ctx.Input<phi::DenseTensor>("OutLinearBias");
    auto *qkv_weight_data = qkv_weight->data<T>();
    auto *qkv_bias_data = (qkv_bias == nullptr) ? nullptr : qkv_bias->data<T>();
    auto *out_linear_weight_data = out_linear_weight->data<T>();
    auto *out_linear_bias_data =
        (out_linear_bias == nullptr) ? nullptr : out_linear_bias->data<T>();

    // fw output
    auto *fmha_out = ctx.Input<phi::DenseTensor>("FMHAOut");
    auto *transpose_out_2 = ctx.Input<phi::DenseTensor>("TransposeOut2");
    auto *qk_out = ctx.Input<phi::DenseTensor>("QKOut");
    auto *softmax_out = ctx.Input<phi::DenseTensor>("SoftmaxOut");
    auto *attn_dropout_mask_out =
        ctx.Input<phi::DenseTensor>("AttnDropoutMaskOut");
    auto *attn_dropout_out = ctx.Input<phi::DenseTensor>("AttnDropoutOut");
    auto *src_mask_out = ctx.Input<phi::DenseTensor>("SrcMaskOut");
    auto *ln_2_mean = ctx.Input<phi::DenseTensor>("Ln2Mean");
    auto *ln_2_var = ctx.Input<phi::DenseTensor>("Ln2Variance");
    auto *dropout_mask_out = ctx.Input<phi::DenseTensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Input<phi::DenseTensor>("BiasDropoutResidualOut");
    auto *fmha_out_data = fmha_out->data<T>();
    auto *transpose_out_2_data = transpose_out_2->data<T>();
    auto *softmax_out_data = softmax_out->data<T>();
    auto *src_mask_out_data =
        (src_mask == nullptr) ? nullptr : src_mask_out->data<T>();
    auto *dropout_mask_out_data =
        has_dropout ? dropout_mask_out->data<uint8_t>() : nullptr;

    // output's grad
    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_qkv_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVOut"));
    auto *d_qkv_bias_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVBiasOut"));
    auto *d_qktv_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKTVOut"));
    auto *d_transpose_out_2 =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("TransposeOut2"));
    auto *d_qk_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKOut"));
    auto *d_softmax_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("SoftmaxOut"));
    auto *d_attn_dropout_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("AttnDropoutOut"));
    auto *d_src_mask_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("SrcMaskOut"));
    auto *d_fmha_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("FMHAOut"));
    auto *d_out_linear_out =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearOut"));
    auto *d_bias_dropout_residual_out = ctx.Output<phi::DenseTensor>(
        framework::GradVarName("BiasDropoutResidualOut"));
    auto *d_x_data = dev_ctx.template Alloc<T>(d_x, d_x->numel() * sizeof(T));
    // when qkv_bias is not nullptr, d_qkv_out is equals to d_qkv_bias_out, the
    // space can be reused.
    auto *d_qkv_out_data = (d_qkv_bias_out != nullptr)
                               ? nullptr
                               : dev_ctx.template Alloc<T>(
                                     d_qkv_out, d_qkv_out->numel() * sizeof(T));
    auto *d_qkv_bias_out_data =
        (d_qkv_bias_out == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(d_qkv_bias_out,
                                        d_qkv_bias_out->numel() * sizeof(T));
    auto *d_qktv_out_data =
        dev_ctx.template Alloc<T>(d_qktv_out, d_qktv_out->numel() * sizeof(T));
    auto *d_transpose_out_2_data = dev_ctx.template Alloc<T>(
        d_transpose_out_2, d_transpose_out_2->numel() * sizeof(T));
    auto *d_qk_out_data =
        dev_ctx.template Alloc<T>(d_qk_out, d_qk_out->numel() * sizeof(T));
    auto *d_softmax_out_data = dev_ctx.template Alloc<T>(
        d_softmax_out, d_softmax_out->numel() * sizeof(T));
    auto *d_attn_dropout_out_data =
        has_attn_dropout
            ? dev_ctx.template Alloc<T>(d_attn_dropout_out,
                                        d_attn_dropout_out->numel() * sizeof(T))
            : nullptr;
    auto *d_src_mask_out_data =
        (src_mask == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(d_src_mask_out,
                                        d_src_mask_out->numel() * sizeof(T));
    auto *d_fmha_out_data =
        dev_ctx.template Alloc<T>(d_fmha_out, d_fmha_out->numel() * sizeof(T));
    auto *d_out_linear_out_data = dev_ctx.template Alloc<T>(
        d_out_linear_out, d_out_linear_out->numel() * sizeof(T));

    // parameter grad
    auto *d_qkv_weight =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVW"));
    auto *d_qkv_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("QKVBias"));
    auto *d_out_linear_weight =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearW"));
    auto *d_out_linear_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("OutLinearBias"));
    auto *d_ln_2_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Ln2Scale"));
    auto *d_ln_2_bias =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Ln2Bias"));

    auto *d_qkv_weight_data = dev_ctx.template Alloc<T>(
        d_qkv_weight, d_qkv_weight->numel() * sizeof(T));
    auto *d_qkv_bias_data =
        (d_qkv_bias == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(d_qkv_bias,
                                        d_qkv_bias->numel() * sizeof(T));
    auto *d_out_linear_weight_data = dev_ctx.template Alloc<T>(
        d_out_linear_weight, d_out_linear_weight->numel() * sizeof(T));
    auto *d_out_linear_bias_data =
        (d_out_linear_bias == nullptr)
            ? nullptr
            : dev_ctx.template Alloc<T>(d_out_linear_bias,
                                        d_out_linear_bias->numel() * sizeof(T));

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

    bool add_residual = ctx.Attr<bool>("add_residual");
    phi::DenseTensor d_residual;
    T *d_residual_data = nullptr;
    if (add_residual) {
      d_residual.Resize(input_x_dims);
      d_residual_data = dev_ctx.template Alloc<T>(
          &d_residual, d_residual.numel() * sizeof(T));
    }

    bool transA = false;
    bool transB = true;
    bool compute_qkv_bias = qkv_bias ? true : false;
    auto layer_norm_compute = AttnLayerNorm<T>(
        ctx.cuda_device_context(), epsilon, bsz_seq, dim_embed);
    auto qkv_compute = AttnMatMul<T>(ctx.cuda_device_context(),
                                     transA,
                                     transB,
                                     bsz_seq,
                                     output_size,
                                     input_size,
                                     compute_qkv_bias);
    AttnDropoutParam attn_dropout_param(is_test_1,
                                        dropout_implementation_1,
                                        attn_dropout_prob,
                                        is_upscale_in_train_1,
                                        is_fix_seed_1,
                                        seed_val_1,
                                        seed_1);
    auto fmha_ref_compute = FMHARef<T>(ctx.cuda_device_context(),
                                       batch_size,
                                       max_seq_len,
                                       num_head,
                                       dim_head,
                                       attn_dropout_param);
    output_size = hidden_size;
    transA = false;
    transB = false;
    bool compute_bias = false;
    // (b*s, num_head * dim_head) * (num_head * dim_head, dim_embed)
    auto out_linear_compute = AttnMatMul<T>(ctx.cuda_device_context(),
                                            transA,
                                            transB,
                                            bsz_seq,
                                            input_size,
                                            output_size,
                                            compute_bias);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        ctx.cuda_device_context(),
        bsz_seq,
        dim_embed,
        dropout_param2,
        ln2epsilon);

    if (pre_layer_norm) {
      fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
          ctx.cuda_device_context(),
          d_y_data,
          dropout_mask_out_data,
          d_out_linear_out_data,
          d_residual_data,
          d_out_linear_bias_data);
    } else {
      auto *ln_2_mean_data = ln_2_mean->data<U>();
      auto *ln_2_var_data = ln_2_var->data<U>();
      auto *bias_dropout_residual_out_data =
          bias_dropout_residual_out->data<T>();
      auto *d_ln_2_scale_data =
          (d_ln_2_scale == nullptr
               ? nullptr
               : dev_ctx.template Alloc<U>(d_ln_2_scale,
                                           d_ln_2_scale->numel() * sizeof(U)));
      auto *d_ln_2_bias_data =
          (d_ln_2_bias == nullptr
               ? nullptr
               : dev_ctx.template Alloc<U>(d_ln_2_bias,
                                           d_ln_2_bias->numel() * sizeof(U)));
      auto *d_bias_dropout_residual_out_data = dev_ctx.template Alloc<T>(
          d_bias_dropout_residual_out,
          d_bias_dropout_residual_out->numel() * sizeof(T));

      fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
          ctx.cuda_device_context(),
          d_y_data,
          bias_dropout_residual_out_data,
          dropout_mask_out_data,
          ln_2_scale_data,
          ln_2_mean_data,
          ln_2_var_data,
          d_bias_dropout_residual_out_data,
          d_ln_2_scale_data,
          d_ln_2_bias_data,
          d_out_linear_out_data,
          d_out_linear_bias_data,
          d_residual_data);
    }

    out_linear_compute.ComputeBackward(fmha_out,
                                       out_linear_weight,
                                       d_out_linear_out,
                                       d_fmha_out,
                                       d_out_linear_weight,
                                       nullptr);

    if (qkv_bias != nullptr) {
      fmha_ref_compute.ComputeBackward(*transpose_out_2,
                                       has_attn_dropout ? src_mask : nullptr,
                                       *softmax_out,
                                       *attn_dropout_mask_out,
                                       *attn_dropout_out,
                                       *qk_out,
                                       *src_mask_out,
                                       *d_fmha_out,
                                       d_qktv_out,
                                       d_attn_dropout_out,
                                       d_softmax_out,
                                       d_src_mask_out,
                                       d_qk_out,
                                       d_transpose_out_2,
                                       nullptr,
                                       d_qkv_bias_out);
    } else {
      fmha_ref_compute.ComputeBackward(*transpose_out_2,
                                       has_attn_dropout ? src_mask : nullptr,
                                       *softmax_out,
                                       *attn_dropout_mask_out,
                                       *attn_dropout_out,
                                       *qk_out,
                                       *src_mask_out,
                                       *d_fmha_out,
                                       d_qktv_out,
                                       d_attn_dropout_out,
                                       d_softmax_out,
                                       d_src_mask_out,
                                       d_qk_out,
                                       d_transpose_out_2,
                                       nullptr,
                                       d_qkv_out);
    }

    if (pre_layer_norm) {
      auto *ln_mean = ctx.Input<phi::DenseTensor>("LnMean");
      auto *ln_var = ctx.Input<phi::DenseTensor>("LnVariance");
      auto *ln_out = ctx.Input<phi::DenseTensor>("LnOut");
      auto *ln_mean_data = ln_mean->data<U>();
      auto *ln_var_data = ln_var->data<U>();
      auto *ln_out_data = ln_out->data<T>();

      auto *d_ln_out =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("LnOut"));
      auto *d_ln_scale =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("LnScale"));
      auto *d_ln_bias =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("LnBias"));
      auto *d_ln_out_data =
          dev_ctx.template Alloc<T>(d_ln_out, d_ln_out->numel() * sizeof(T));
      auto *d_ln_scale_data =
          (d_ln_scale == nullptr
               ? nullptr
               : dev_ctx.template Alloc<U>(d_ln_scale,
                                           d_ln_scale->numel() * sizeof(U)));
      auto *d_ln_bias_data =
          (d_ln_bias == nullptr
               ? nullptr
               : dev_ctx.template Alloc<U>(d_ln_bias,
                                           d_ln_bias->numel() * sizeof(U)));
      if (qkv_bias != nullptr) {
        qkv_compute.ComputeBackward(ln_out,
                                    qkv_weight,
                                    d_qkv_bias_out,
                                    d_ln_out,
                                    d_qkv_weight,
                                    d_qkv_bias);
      } else {
        qkv_compute.ComputeBackward(
            ln_out, qkv_weight, d_qkv_out, d_ln_out, d_qkv_weight, d_qkv_bias);
      }
      // tensor model parallel
      AllReduce<T>(*d_ln_out, ring_id, ctx.cuda_device_context());
      layer_norm_compute.ComputeBackward(x_data,
                                         d_ln_out_data,
                                         ln_scale_data,
                                         ln_mean_data,
                                         ln_var_data,
                                         d_x_data,
                                         d_ln_scale_data,
                                         d_ln_bias_data);
    } else {
      if (qkv_bias != nullptr) {
        qkv_compute.ComputeBackward(
            input_x, qkv_weight, d_qkv_bias_out, d_x, d_qkv_weight, d_qkv_bias);
      } else {
        qkv_compute.ComputeBackward(
            input_x, qkv_weight, d_qkv_out, d_x, d_qkv_weight, d_qkv_bias);
      }
      // tensor model parallel
      AllReduce<T>(*d_x, ring_id, ctx.cuda_device_context());
    }

    if (add_residual) {
      // gradient accumulation
      std::vector<const phi::DenseTensor *> ins = {&d_residual, d_x};
      std::vector<phi::DenseTensor *> outs = {d_x};
      phi::funcs::ElementwiseKernel<T>(
          ctx.cuda_device_context(), ins, &outs, phi::funcs::AddFunctor<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_attention,
                        ops::FusedAttentionOpKernel<float>,
                        ops::FusedAttentionOpKernel<double>,
                        ops::FusedAttentionOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(fused_attention_grad,
                        ops::FusedAttentionGradKernel<float>,
                        ops::FusedAttentionGradKernel<double>,
                        ops::FusedAttentionGradKernel<plat::float16>);
