// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#include "paddle/phi/kernels/fusion/gpu/fmha_ref.h"
#include "paddle/phi/kernels/fusion/gpu/fused_attention_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedAttentionGradKernel(
    const Context &dev_ctx,
    const DenseTensor &out_grad,
    const DenseTensor &x,
    const DenseTensor &qkv_weight,
    const paddle::optional<DenseTensor> &qkv_bias,
    const paddle::optional<DenseTensor> &qkv_bias_out,
    const paddle::optional<DenseTensor> &src_mask,
    const paddle::optional<DenseTensor> &src_mask_out,
    const DenseTensor &out_linear_weight,
    const paddle::optional<DenseTensor> &out_linear_bias,
    const paddle::optional<DenseTensor> &ln_scale,
    const paddle::optional<DenseTensor> &ln_bias,
    const paddle::optional<DenseTensor> &ln_scale_2,
    const paddle::optional<DenseTensor> &ln_bias_2,
    const paddle::optional<DenseTensor> &ln_out,
    const paddle::optional<DenseTensor> &ln_mean,
    const paddle::optional<DenseTensor> &ln_var,
    const paddle::optional<DenseTensor> &ln_mean_2,
    const paddle::optional<DenseTensor> &ln_var_2,
    const paddle::optional<DenseTensor> &bias_dropout_residual_out,
    const DenseTensor &qkv_out,
    const DenseTensor &transpose_out_2,
    const DenseTensor &qk_out,
    const DenseTensor &qktv_out,
    const DenseTensor &softmax_out,
    const DenseTensor &attn_dropout_mask_out,
    const DenseTensor &attn_dropout_out,
    const DenseTensor &fmha_out,
    const DenseTensor &out_linear_out,
    const DenseTensor &dropout_mask_out,
    int num_heads,
    bool transpose_qkv_wb,
    bool pre_layer_norm,
    float epsilon,
    float attn_dropout_rate,
    bool is_test,
    bool attn_dropout_fix_seed,
    int attn_dropout_seed,
    const std::string &attn_dropout_implementation,
    float dropout_rate,
    bool dropout_fix_seed,
    int dropout_seed,
    const std::string &dropout_implementation,
    float ln_epsilon,
    bool add_residual,
    int ring_id,
    DenseTensor *qkv_bias_grad,
    DenseTensor *qkv_bias_out_grad,
    DenseTensor *src_mask_out_grad,
    DenseTensor *out_linear_bias_grad,
    DenseTensor *ln_scale_grad,
    DenseTensor *ln_bias_grad,
    DenseTensor *ln_scale_2_grad,
    DenseTensor *ln_bias_2_grad,
    DenseTensor *x_grad,
    DenseTensor *qkv_weight_grad,
    DenseTensor *out_linear_weight_grad,
    DenseTensor *ln_out_grad,
    DenseTensor *bias_dropout_residual_out_grad,
    DenseTensor *qkv_out_grad,
    DenseTensor *qktv_out_grad,
    DenseTensor *transpose_out_2_grad,
    DenseTensor *qk_out_grad,
    DenseTensor *softmax_out_grad,
    DenseTensor *attn_dropout_out_grad,
    DenseTensor *fmha_out_grad,
    DenseTensor *out_linear_out_grad) {
  using U = phi::fusion::LayerNormParamType<T>;

  const bool has_attn_dropout = (attn_dropout_rate != 0.0f);

  const bool is_upscale_in_train =
      (dropout_implementation == "upscale_in_train");
  phi::fusion::DropoutParam dropout_param2(dropout_fix_seed,
                                           0,
                                           is_test,
                                           is_upscale_in_train,
                                           dropout_rate,
                                           nullptr,
                                           dropout_seed);
  const bool has_dropout = (dropout_param2.dropout_prob != 0.0f);

  bool is_upscale_in_train_1 =
      (attn_dropout_implementation == "upscale_in_train");
  phi::DenseTensor *seed_1 = nullptr;

  // get inputs.
  auto *d_y = &out_grad;
  auto *d_y_data = d_y->data<T>();

  // fw input
  auto *input_x = &x;
  auto *ln_scale_p = ln_scale.get_ptr();
  auto *ln_scale_2_p = ln_scale_2.get_ptr();
  auto *x_data = input_x->data<T>();
  auto *ln_scale_data =
      (ln_scale_p == nullptr ? nullptr : ln_scale_p->data<U>());
  auto *ln_2_scale_data =
      (ln_scale_2_p == nullptr ? nullptr : ln_scale_2_p->data<U>());
  // fw parameters.
  auto *src_mask_p = src_mask.get_ptr();
  auto *qkv_weight_p = &qkv_weight;
  auto *qkv_bias_p = qkv_bias.get_ptr();
  auto *out_linear_weight_p = &out_linear_weight;
  auto *out_linear_bias_p = out_linear_bias.get_ptr();
  auto *qkv_weight_data = qkv_weight_p->data<T>();
  auto *qkv_bias_data =
      (qkv_bias_p == nullptr) ? nullptr : qkv_bias_p->data<T>();
  auto *out_linear_weight_data = out_linear_weight_p->data<T>();
  auto *out_linear_bias_data =
      (out_linear_bias_p == nullptr) ? nullptr : out_linear_bias_p->data<T>();

  // fw output
  auto *fmha_out_p = &fmha_out;
  auto *transpose_out_2_p = &transpose_out_2;
  auto *qk_out_p = &qk_out;
  auto *softmax_out_p = &softmax_out;
  auto *attn_dropout_mask_out_p = &attn_dropout_mask_out;
  auto *attn_dropout_out_p = &attn_dropout_out;
  auto *src_mask_out_p = src_mask_out.get_ptr();
  auto *ln_mean_2_p = ln_mean_2.get_ptr();
  auto *ln_var_2_p = ln_var_2.get_ptr();
  auto *dropout_mask_out_p = &dropout_mask_out;
  auto *bias_dropout_residual_out_p = bias_dropout_residual_out.get_ptr();
  auto *fmha_out_data = fmha_out_p->data<T>();
  auto *transpose_out_2_data = transpose_out_2_p->data<T>();
  auto *softmax_out_data = softmax_out_p->data<T>();
  auto *src_mask_out_data =
      (src_mask_p == nullptr) ? nullptr : src_mask_out_p->data<T>();
  auto *dropout_mask_out_data =
      has_dropout ? dropout_mask_out_p->data<uint8_t>() : nullptr;

  auto *d_x_data =
      dev_ctx.template Alloc<T>(x_grad, x_grad->numel() * sizeof(T));
  // when qkv_bias_p is not nullptr, qkv_out_grad is equals to
  // qkv_bias_out_grad, the space can be reused.
  auto *d_qkv_out_data =
      (qkv_bias_out_grad != nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(qkv_out_grad,
                                      qkv_out_grad->numel() * sizeof(T));
  auto *d_qkv_bias_out_data =
      (qkv_bias_out_grad == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(qkv_bias_out_grad,
                                      qkv_bias_out_grad->numel() * sizeof(T));
  auto *d_qktv_out_data = dev_ctx.template Alloc<T>(
      qktv_out_grad, qktv_out_grad->numel() * sizeof(T));
  auto *d_transpose_out_2_data = dev_ctx.template Alloc<T>(
      transpose_out_2_grad, transpose_out_2_grad->numel() * sizeof(T));
  auto *d_qk_out_data =
      dev_ctx.template Alloc<T>(qk_out_grad, qk_out_grad->numel() * sizeof(T));
  auto *d_softmax_out_data = dev_ctx.template Alloc<T>(
      softmax_out_grad, softmax_out_grad->numel() * sizeof(T));
  auto *d_attn_dropout_out_data =
      has_attn_dropout ? dev_ctx.template Alloc<T>(
                             attn_dropout_out_grad,
                             attn_dropout_out_grad->numel() * sizeof(T))
                       : nullptr;
  auto *d_src_mask_out_data =
      (src_mask_p == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(src_mask_out_grad,
                                      src_mask_out_grad->numel() * sizeof(T));
  auto *d_fmha_out_data = dev_ctx.template Alloc<T>(
      fmha_out_grad, fmha_out_grad->numel() * sizeof(T));
  auto *d_out_linear_out_data = dev_ctx.template Alloc<T>(
      out_linear_out_grad, out_linear_out_grad->numel() * sizeof(T));

  // parameter grad
  auto *d_qkv_weight_data =
      (qkv_weight_grad == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(qkv_weight_grad,
                                      qkv_weight_grad->numel() * sizeof(T));

  auto *d_qkv_bias_data =
      (qkv_bias_grad == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(qkv_bias_grad,
                                      qkv_bias_grad->numel() * sizeof(T));
  auto *d_out_linear_weight_data =
      (out_linear_weight_grad == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(
                out_linear_weight_grad,
                out_linear_weight_grad->numel() * sizeof(T));

  auto *d_out_linear_bias_data =
      (out_linear_bias_grad == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<T>(
                out_linear_bias_grad,
                out_linear_bias_grad->numel() * sizeof(T));

  const auto input_x_dims = input_x->dims();
  const auto qkv_w_dims = qkv_weight_p->dims();

  int batch_size = input_x_dims[0];
  int max_seq_len = input_x_dims[1];
  int dim_embed = input_x_dims[2];
  int num_head;
  int dim_head;
  int nranks = 1;
  if (!transpose_qkv_wb) {
    num_head = qkv_w_dims[1];
    dim_head = qkv_w_dims[2];
  } else {
    nranks = (qkv_w_dims[0] * 3) / qkv_w_dims[1];
    num_head = num_heads;
    dim_head = dim_embed / (num_head * nranks);
  }

  int bsz_seq = batch_size * max_seq_len;
  int hidden_size = num_head * dim_head;
  int output_size = 3 * hidden_size;
  int input_size = dim_embed;

  phi::DenseTensor d_residual;
  T *d_residual_data = nullptr;
  if (add_residual) {
    d_residual.Resize(input_x_dims);
    d_residual_data =
        dev_ctx.template Alloc<T>(&d_residual, d_residual.numel() * sizeof(T));
  }

  bool transA = false;
  bool transB = transpose_qkv_wb ? false : true;
  bool compute_qkv_bias = qkv_bias_p ? true : false;
  auto layer_norm_compute =
      phi::fusion::AttnLayerNorm<T>(dev_ctx, epsilon, bsz_seq, dim_embed);
  auto qkv_compute = phi::fusion::AttnMatMul<T>(dev_ctx,
                                                transA,
                                                transB,
                                                bsz_seq,
                                                output_size,
                                                input_size,
                                                compute_qkv_bias);
  phi::fusion::AttnDropoutParam attn_dropout_param(is_test,
                                                   attn_dropout_implementation,
                                                   attn_dropout_rate,
                                                   is_upscale_in_train_1,
                                                   attn_dropout_fix_seed,
                                                   attn_dropout_seed,
                                                   seed_1);
  auto fmha_ref_compute = phi::fusion::FMHARef<T>(
      dev_ctx, batch_size, max_seq_len, num_head, dim_head, attn_dropout_param);
  output_size = hidden_size;
  transA = false;
  transB = false;
  bool compute_bias = false;
  // (b*s, num_head * dim_head) * (num_head * dim_head, dim_embed)
  auto out_linear_compute = phi::fusion::AttnMatMul<T>(
      dev_ctx, transA, transB, bsz_seq, input_size, output_size, compute_bias);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper(
          dev_ctx, bsz_seq, dim_embed, dropout_param2, ln_epsilon);

  if (pre_layer_norm) {
    fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
        dev_ctx,
        d_y_data,
        dropout_mask_out_data,
        d_out_linear_out_data,
        d_residual_data,
        d_out_linear_bias_data);
  } else {
    auto *ln_mean_2_data = ln_mean_2_p->data<U>();
    auto *ln_var_2_data = ln_var_2_p->data<U>();
    auto *bias_dropout_residual_out_data =
        bias_dropout_residual_out_p->data<T>();
    auto *d_ln_2_scale_data =
        (ln_scale_2_grad == nullptr
             ? nullptr
             : dev_ctx.template Alloc<U>(ln_scale_2_grad,
                                         ln_scale_2_grad->numel() * sizeof(U)));
    auto *d_ln_bias_2_data =
        (ln_bias_2_grad == nullptr
             ? nullptr
             : dev_ctx.template Alloc<U>(ln_bias_2_grad,
                                         ln_bias_2_grad->numel() * sizeof(U)));
    auto *d_bias_dropout_residual_out_data = dev_ctx.template Alloc<T>(
        bias_dropout_residual_out_grad,
        bias_dropout_residual_out_grad->numel() * sizeof(T));

    fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
        dev_ctx,
        d_y_data,
        bias_dropout_residual_out_data,
        dropout_mask_out_data,
        ln_2_scale_data,
        ln_mean_2_data,
        ln_var_2_data,
        d_bias_dropout_residual_out_data,
        d_ln_2_scale_data,
        d_ln_bias_2_data,
        d_out_linear_out_data,
        d_out_linear_bias_data,
        d_residual_data);
  }

  out_linear_compute.ComputeBackward(fmha_out_p,
                                     out_linear_weight_p,
                                     out_linear_out_grad,
                                     fmha_out_grad,
                                     out_linear_weight_grad,
                                     nullptr);

  if (transpose_qkv_wb) {
    if (compute_qkv_bias) {
      qkv_bias_out_grad->Resize(
          {batch_size, max_seq_len, 3, num_head, dim_head});
    } else {
      qkv_out_grad->Resize({batch_size, max_seq_len, 3, num_head, dim_head});
    }
  }

  if (qkv_bias_p != nullptr) {
    fmha_ref_compute.ComputeBackward(*transpose_out_2_p,
                                     has_attn_dropout ? src_mask_p : nullptr,
                                     *softmax_out_p,
                                     *attn_dropout_mask_out_p,
                                     *attn_dropout_out_p,
                                     *qk_out_p,
                                     *src_mask_out_p,
                                     *fmha_out_grad,
                                     qktv_out_grad,
                                     attn_dropout_out_grad,
                                     softmax_out_grad,
                                     src_mask_out_grad,
                                     qk_out_grad,
                                     transpose_out_2_grad,
                                     nullptr,
                                     qkv_bias_out_grad);
  } else {
    fmha_ref_compute.ComputeBackward(*transpose_out_2_p,
                                     has_attn_dropout ? src_mask_p : nullptr,
                                     *softmax_out_p,
                                     *attn_dropout_mask_out_p,
                                     *attn_dropout_out_p,
                                     *qk_out_p,
                                     *src_mask_out_p,
                                     *fmha_out_grad,
                                     qktv_out_grad,
                                     attn_dropout_out_grad,
                                     softmax_out_grad,
                                     src_mask_out_grad,
                                     qk_out_grad,
                                     transpose_out_2_grad,
                                     nullptr,
                                     qkv_out_grad);
  }

  if (transpose_qkv_wb) {
    if (compute_qkv_bias) {
      qkv_bias_out_grad->Resize({batch_size, max_seq_len, 3 * hidden_size});
    } else {
      qkv_out_grad->Resize({batch_size, max_seq_len, 3 * hidden_size});
    }
  }

  if (pre_layer_norm) {
    auto *ln_mean_p = ln_mean.get_ptr();
    auto *ln_var_p = ln_var.get_ptr();
    auto *ln_out_p = ln_out.get_ptr();
    auto *ln_mean_data = ln_mean_p->data<U>();
    auto *ln_var_data = ln_var_p->data<U>();
    auto *ln_out_data = ln_out_p->data<T>();

    auto *d_ln_out_data = dev_ctx.template Alloc<T>(
        ln_out_grad, ln_out_grad->numel() * sizeof(T));
    auto *d_ln_scale_data =
        (ln_scale_grad == nullptr
             ? nullptr
             : dev_ctx.template Alloc<U>(ln_scale_grad,
                                         ln_scale_grad->numel() * sizeof(U)));
    auto *d_ln_bias_data =
        (ln_bias_grad == nullptr
             ? nullptr
             : dev_ctx.template Alloc<U>(ln_bias_grad,
                                         ln_bias_grad->numel() * sizeof(U)));
    if (qkv_bias_p != nullptr) {
      qkv_compute.ComputeBackward(ln_out_p,
                                  qkv_weight_p,
                                  qkv_bias_out_grad,
                                  ln_out_grad,
                                  qkv_weight_grad,
                                  qkv_bias_grad);
    } else {
      qkv_compute.ComputeBackward(ln_out_p,
                                  qkv_weight_p,
                                  qkv_out_grad,
                                  ln_out_grad,
                                  qkv_weight_grad,
                                  qkv_bias_grad);
    }
    // tensor model parallel
    phi::fusion::AllReduce<T>(*ln_out_grad, ring_id, dev_ctx);
    layer_norm_compute.ComputeBackward(x_data,
                                       d_ln_out_data,
                                       ln_scale_data,
                                       ln_mean_data,
                                       ln_var_data,
                                       d_x_data,
                                       d_ln_scale_data,
                                       d_ln_bias_data);
  } else {
    if (qkv_bias_p != nullptr) {
      qkv_compute.ComputeBackward(input_x,
                                  qkv_weight_p,
                                  qkv_bias_out_grad,
                                  x_grad,
                                  qkv_weight_grad,
                                  qkv_bias_grad);
    } else {
      qkv_compute.ComputeBackward(input_x,
                                  qkv_weight_p,
                                  qkv_out_grad,
                                  x_grad,
                                  qkv_weight_grad,
                                  qkv_bias_grad);
    }
    // tensor model parallel
    phi::fusion::AllReduce<T>(*x_grad, ring_id, dev_ctx);
  }

  if (add_residual) {
    // gradient accumulation
    std::vector<const phi::DenseTensor *> ins = {&d_residual, x_grad};
    std::vector<phi::DenseTensor *> outs = {x_grad};
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, ins, &outs, phi::funcs::AddFunctor<T>());
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedAttentionGradKernel,
                   phi::dtype::float16,
                   double,
                   float) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(7).SetDataType(phi::DataType::FLOAT32);
  }
}
