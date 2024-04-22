// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fused_attention_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#include "paddle/phi/kernels/xpu/xpu_fused_common_function.h"

namespace phi {

template <typename T, typename Context>
void FusedAttentionKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const paddle::optional<DenseTensor> &ln_scale,
                          const paddle::optional<DenseTensor> &ln_bias,
                          const DenseTensor &qkv_weight,
                          const paddle::optional<DenseTensor> &qkv_bias,
                          const paddle::optional<DenseTensor> &cache_kv,
                          const paddle::optional<DenseTensor> &src_mask,
                          const DenseTensor &out_linear_weight,
                          const paddle::optional<DenseTensor> &out_linear_bias,
                          const paddle::optional<DenseTensor> &ln_scale_2,
                          const paddle::optional<DenseTensor> &ln_bias_2,
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
                          DenseTensor *ln_mean,
                          DenseTensor *ln_var,
                          DenseTensor *ln_out,
                          DenseTensor *qkv_out,
                          DenseTensor *qkv_bias_out,
                          DenseTensor *transpose_out_2,
                          DenseTensor *qk_out,
                          DenseTensor *qktv_out,
                          DenseTensor *softmax_out,
                          DenseTensor *attn_dropout_mask_out,
                          DenseTensor *attn_dropout_out,
                          DenseTensor *src_mask_out,
                          DenseTensor *fmha_out,
                          DenseTensor *out_linear_out,
                          DenseTensor *dropout_mask_out,
                          DenseTensor *ln_mean_2,
                          DenseTensor *ln_var_2,
                          DenseTensor *bias_dropout_residual_out,
                          DenseTensor *cache_kv_out,
                          DenseTensor *out) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  // shape [batch_size, 1, 1, seq_len]
  const phi::DenseTensor *src_mask_p = src_mask.get_ptr();

  const phi::DenseTensor *ln_scale_p = nullptr;
  const phi::DenseTensor *ln_bias_p = nullptr;

  if (pre_layer_norm) {
    ln_scale_p = ln_scale.get_ptr();
    ln_bias_p = ln_bias.get_ptr();
  } else {
    ln_scale_p = ln_scale_2.get_ptr();
    ln_bias_p = ln_bias_2.get_ptr();
    epsilon = ln_epsilon;
  }

  dev_ctx.template Alloc<T>(qk_out);
  dev_ctx.template Alloc<T>(qktv_out);
  dev_ctx.template Alloc<T>(out_linear_out);
  dev_ctx.template Alloc<T>(qkv_bias_out);
  dev_ctx.template Alloc<T>(src_mask_out);
  dev_ctx.template Alloc<T>(qkv_out);

  bool is_upscale_in_train_1 =
      (attn_dropout_implementation == "upscale_in_train");
  const phi::DenseTensor *seed_1 = nullptr;

  phi::XPUDropoutParam attn_dropout_param;
  attn_dropout_param.initXPUDropoutParam(attn_dropout_rate,
                                         is_upscale_in_train_1,
                                         is_test,
                                         attn_dropout_fix_seed,
                                         seed_1,
                                         attn_dropout_seed);

  phi::XPUDropoutParam dropout_param;
  dropout_param.initXPUDropoutParam(dropout_rate,
                                    is_upscale_in_train_1,
                                    is_test,
                                    dropout_fix_seed,
                                    seed_1,
                                    dropout_seed);

  // 先计算纬度
  const auto input_x_dims = x.dims();
  const auto qkv_w_dims = qkv_weight.dims();

  int batch_size = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int embed_dims = input_x_dims[2];
  num_heads = qkv_w_dims[1];
  int head_dims = qkv_w_dims[2];

  // 输入指针
  const XPUTypeT *input_x_ptr = reinterpret_cast<const XPUTypeT *>(x.data<T>());

  const XPUTypeT *qkv_weight_ptr =
      reinterpret_cast<const XPUTypeT *>(qkv_weight.data<T>());
  const DenseTensor *qkv_bias_p = qkv_bias.get_ptr();
  const XPUTypeT *qkv_bias_ptr =
      reinterpret_cast<const XPUTypeT *>(qkv_bias_p->data<T>());
  const XPUTypeT *src_mask_ptr =
      (src_mask_p == nullptr)
          ? (nullptr)
          : (reinterpret_cast<const XPUTypeT *>(src_mask_p->data<T>()));

  const XPUTypeT *out_linear_weight_ptr =
      reinterpret_cast<const XPUTypeT *>(out_linear_weight.data<T>());

  const DenseTensor *out_linear_bias_p = out_linear_bias.get_ptr();
  const XPUTypeT *out_linear_bias_ptr =
      reinterpret_cast<const XPUTypeT *>(out_linear_bias_p->data<T>());

  const float *ln_scale_ptr =
      (ln_scale_p == nullptr) ? (nullptr) : ln_scale_p->data<float>();

  const float *ln_bias_ptr =
      (ln_bias_p == nullptr) ? (nullptr) : ln_bias_p->data<float>();

  // 输出指针
  XPUTypeT *qkv_transpose_out_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(transpose_out_2));

  XPUTypeT *softmax_out_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(softmax_out));

  XPUTypeT *attn_dropout_mask_out_ptr = reinterpret_cast<XPUTypeT *>(
      dev_ctx.template Alloc<T>(attn_dropout_mask_out));

  XPUTypeT *attn_dropout_out_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(attn_dropout_out));

  XPUTypeT *fmha_out_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(fmha_out));

  XPUTypeT *dropout_mask_out_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(dropout_mask_out));

  XPUTypeT *out_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(out));

  XPUTypeT *bias_dropout_residual_out_ptr =
      (bias_dropout_residual_out == nullptr)
          ? (nullptr)
          : (reinterpret_cast<XPUTypeT *>(
                dev_ctx.template Alloc<T>(bias_dropout_residual_out)));

  float *ln_mean_ptr =
      (ln_mean == nullptr)
          ? (nullptr)
          : reinterpret_cast<float *>(dev_ctx.template Alloc<float>(ln_mean));

  float *ln_var_ptr =
      (ln_var == nullptr)
          ? (nullptr)
          : reinterpret_cast<float *>(dev_ctx.template Alloc<float>(ln_var));

  XPUTypeT *ln_out_ptr =
      (ln_out == nullptr)
          ? (nullptr)
          : (reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(ln_out)));

  xpu::Context *xpu_ctx = dev_ctx.x_context();

  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  int l3_total_size = xpu_ctx->_l3_mgr.get_size();

  XPUTypeT *qkv_before_transpose_ptr =
      NULL;                  // x2[batch_size, seq_len, 3, num_heads,head_dims]
  XPUTypeT *qk_ptr = NULL;   // qk [batch_size, num_heads, seq_len, seq_len]
  XPUTypeT *qkv_ptr = NULL;  // qkv[batch_size, num_heads, seq_len, head_dims]
  XPUTypeT *linear_out_ptr = NULL;  // x4, x5 [batch_size, seq_len, embed_dims]

  int temp_size_1 = batch_size * seq_len * 3 * num_heads * head_dims;
  int temp_size_2 = batch_size * num_heads * seq_len * seq_len;
  int temp_size_3 = batch_size * num_heads * seq_len * head_dims;
  int temp_size_4 = batch_size * seq_len * embed_dims;

  std::vector<int> temp_vec = {
      temp_size_1, temp_size_2, temp_size_3, temp_size_4};
  std::sort(temp_vec.begin(), temp_vec.end(), std::greater<int>());
  XPUTypeT *max_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(temp_vec[0]);
  PADDLE_ENFORCE_XDNN_NOT_NULL(max_gm_ptr);
  qkv_before_transpose_ptr = max_gm_ptr;
  qk_ptr = max_gm_ptr;
  qkv_ptr = max_gm_ptr;
  linear_out_ptr = max_gm_ptr;
  int sizeof_t = sizeof(XPUTypeT);
  for (size_t i = 0; i < temp_vec.size(); ++i) {
    if (l3_total_size >= temp_vec[i] * sizeof_t) {
      XPUTypeT *l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(temp_vec[i]);
      qkv_before_transpose_ptr =
          (temp_size_1 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
      qk_ptr = (temp_size_2 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
      qkv_ptr = (temp_size_3 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
      linear_out_ptr = (temp_size_4 <= temp_vec[i]) ? l3_ptr : max_gm_ptr;
      break;
    }
  }

  int r = 0;
  const XPUTypeT *x_cacl_ptr = input_x_ptr;
  if (pre_layer_norm) {
    r = xpu::layer_norm(xpu_ctx,
                        input_x_ptr,
                        ln_out_ptr,
                        batch_size * seq_len,
                        embed_dims,
                        epsilon,
                        ln_scale_ptr,
                        ln_bias_ptr,
                        ln_mean_ptr,
                        ln_var_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
    x_cacl_ptr = ln_out_ptr;
  }

  // fc
  phi::XpuFcInfo qkv_fc_info;
  qkv_fc_info.InitFcInfo(0,
                         batch_size * seq_len,
                         3 * num_heads * head_dims,
                         embed_dims,
                         false,
                         true,
                         nullptr,
                         nullptr,
                         nullptr);

  phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                   x_cacl_ptr,
                                   qkv_weight_ptr,
                                   qkv_before_transpose_ptr,
                                   qkv_fc_info,
                                   1.0f);

  // bias
  r = xpu::broadcast_add(xpu_ctx,
                         qkv_before_transpose_ptr,
                         qkv_bias_ptr,
                         qkv_before_transpose_ptr,
                         {batch_size * seq_len, 3 * num_heads * head_dims},
                         {3 * num_heads * head_dims});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  // transpose
  r = xpu::transpose(xpu_ctx,
                     qkv_before_transpose_ptr,
                     qkv_transpose_out_ptr,
                     {batch_size, seq_len, 3, num_heads, head_dims},
                     {2, 0, 3, 1, 4});

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

  int qkv_every_size = batch_size * seq_len * num_heads * head_dims;
  {
    float alpha = 1.0 / sqrt(head_dims);
    r = scale(xpu_ctx,
              qkv_transpose_out_ptr,
              qkv_transpose_out_ptr,
              qkv_every_size,
              false,
              alpha,
              0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }

  // begin fhma
  // 1. qk 2. qk + mask 3. softmax 4.dropout 5. qkv 6. transpos
  {
    const XPUTypeT *q_ptr = qkv_transpose_out_ptr;
    const XPUTypeT *k_ptr = q_ptr + qkv_every_size;
    const XPUTypeT *v_ptr = k_ptr + qkv_every_size;
    phi::XpuFcInfo qk_fc_info;
    qk_fc_info.InitFcInfo(batch_size * num_heads,
                          seq_len,
                          seq_len,
                          head_dims,
                          false,
                          true,
                          nullptr,
                          nullptr,
                          nullptr);
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, q_ptr, k_ptr, qk_ptr, qk_fc_info, 1.0f);

    if (src_mask_ptr) {
      r = xpu::broadcast_add(xpu_ctx,
                             qk_ptr,
                             src_mask_ptr,
                             qk_ptr,
                             {batch_size, num_heads, seq_len, seq_len},
                             {batch_size, 1, 1, seq_len});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
    }
    // do softmax
    r = xpu::softmax(xpu_ctx,
                     qk_ptr,
                     softmax_out_ptr,
                     {batch_size, num_heads, seq_len, seq_len},
                     3);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");

    // do dropout
    phi::Dropout<XPUTypeT>(xpu_ctx,
                           softmax_out_ptr,
                           attn_dropout_mask_out_ptr,
                           attn_dropout_out_ptr,
                           attn_dropout_param,
                           batch_size * num_heads * seq_len * seq_len);

    phi::XpuFcInfo qktv_fc_info;
    qktv_fc_info.InitFcInfo(batch_size * num_heads,
                            seq_len,
                            head_dims,
                            seq_len,
                            false,
                            false,
                            nullptr,
                            nullptr,
                            nullptr);
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, attn_dropout_out_ptr, v_ptr, qkv_ptr, qktv_fc_info, 1.0f);
    r = xpu::transpose(xpu_ctx,
                       qkv_ptr,
                       fmha_out_ptr,
                       {batch_size, num_heads, seq_len, head_dims},
                       {0, 2, 1, 3});

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  }

  // linear_out
  phi::XpuFcInfo linear_fc_info;
  linear_fc_info.InitFcInfo(0,
                            batch_size * seq_len,
                            embed_dims,
                            embed_dims,
                            false,
                            false,
                            nullptr,
                            nullptr,
                            nullptr);
  phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                   fmha_out_ptr,
                                   out_linear_weight_ptr,
                                   linear_out_ptr,
                                   linear_fc_info,
                                   1.0f);

  // out_linear_bias_ptr
  r = xpu::broadcast_add(xpu_ctx,
                         linear_out_ptr,
                         out_linear_bias_ptr,
                         linear_out_ptr,
                         {batch_size * seq_len, embed_dims},
                         {embed_dims});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  Dropout(xpu_ctx,
          linear_out_ptr,
          dropout_mask_out_ptr,
          linear_out_ptr,
          dropout_param,
          batch_size * seq_len * embed_dims);

  XPUTypeT *real_out_ptr = out_ptr;
  if (pre_layer_norm == false) {
    real_out_ptr = bias_dropout_residual_out_ptr;
  }

  r = xpu::add(xpu_ctx,
               linear_out_ptr,
               input_x_ptr,
               real_out_ptr,
               batch_size * seq_len * embed_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");

  if (pre_layer_norm == false) {
    r = xpu::layer_norm(xpu_ctx,
                        real_out_ptr,
                        out_ptr,
                        batch_size * seq_len,
                        embed_dims,
                        epsilon,
                        ln_scale_ptr,
                        ln_bias_ptr,
                        ln_mean_ptr,
                        ln_var_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(fused_attention,
                   XPU,
                   ALL_LAYOUT,
                   phi::FusedAttentionKernel,
                   float,
                   phi::dtype::float16) {}
