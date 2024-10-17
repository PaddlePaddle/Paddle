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

#include "paddle/phi/kernels/fused_attention_grad_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#include "paddle/phi/kernels/xpu/xpu_fused_common_function.h"

namespace phi {

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
    const DenseTensor &attn_dropout_mask,
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
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

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
  // get inputs.
  const XPUTypeT *d_y_ptr =
      reinterpret_cast<const XPUTypeT *>(out_grad.data<T>());
  // 前向必要参数
  const XPUTypeT *input_x_ptr = reinterpret_cast<const XPUTypeT *>(x.data<T>());
  const XPUTypeT *qkv_transpose_out_ptr =
      reinterpret_cast<const XPUTypeT *>(transpose_out_2.data<T>());
  const XPUTypeT *qkv_weight_ptr =
      reinterpret_cast<const XPUTypeT *>(qkv_weight.data<T>());

  const XPUTypeT *softmax_out_ptr =
      reinterpret_cast<const XPUTypeT *>(softmax_out.data<T>());
  const XPUTypeT *attn_dropout_out_ptr =
      reinterpret_cast<const XPUTypeT *>(attn_dropout_out.data<T>());

  const XPUTypeT *attn_dropout_mask_ptr =
      reinterpret_cast<const XPUTypeT *>(attn_dropout_mask.data<T>());
  const XPUTypeT *fmha_out_ptr =
      reinterpret_cast<const XPUTypeT *>(fmha_out.data<T>());

  const XPUTypeT *out_linear_weight_ptr =
      reinterpret_cast<const XPUTypeT *>(out_linear_weight.data<T>());

  const XPUTypeT *dropout_mask_out_ptr =
      reinterpret_cast<const XPUTypeT *>(dropout_mask_out.data<T>());
  // 需要计算的梯度
  auto *d_qkv_weight = qkv_weight_grad;
  XPUTypeT *d_qkv_weight_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(d_qkv_weight));

  auto *d_qkv_bias = qkv_bias_grad;
  XPUTypeT *d_qkv_bias_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(d_qkv_bias));
  auto *d_out_linear_weight = out_linear_weight_grad;

  XPUTypeT *d_out_linear_weight_ptr = reinterpret_cast<XPUTypeT *>(
      dev_ctx.template Alloc<T>(d_out_linear_weight));

  auto *d_out_linear_bias = out_linear_bias_grad;
  XPUTypeT *d_out_linear_bias_ptr = reinterpret_cast<XPUTypeT *>(
      dev_ctx.template Alloc<T>(d_out_linear_bias));
  // 有可能需要
  auto *d_src_mask_out = src_mask_out_grad;
  XPUTypeT *d_src_mask_out_ptr =
      (d_src_mask_out == nullptr)
          ? (nullptr)
          : (reinterpret_cast<XPUTypeT *>(
                dev_ctx.template Alloc<T>(d_src_mask_out)));
  // 输出 dx
  auto *d_x = x_grad;
  XPUTypeT *d_x_ptr =
      reinterpret_cast<XPUTypeT *>(dev_ctx.template Alloc<T>(d_x));

  const phi::DenseTensor *ln_out_p = ln_out.get_ptr();
  const phi::DenseTensor *bias_dropout_residual_out_p =
      bias_dropout_residual_out.get_ptr();

  const phi::DenseTensor *ln_scale_p = nullptr;
  const phi::DenseTensor *ln_mean_p = nullptr;
  const phi::DenseTensor *ln_var_p = nullptr;
  phi::DenseTensor *d_ln_scale = nullptr;
  phi::DenseTensor *d_ln_bias = nullptr;

  const XPUTypeT *ln_out_ptr = NULL;
  const float *ln_scale_ptr = NULL;
  const float *ln_mean_ptr = NULL;
  const float *ln_var_ptr = NULL;
  const XPUTypeT *bias_dropout_residual_out_ptr = NULL;
  float *d_ln_scale_ptr = nullptr;
  float *d_ln_bias_ptr = nullptr;

  if (pre_layer_norm) {
    ln_out_ptr = reinterpret_cast<const XPUTypeT *>(ln_out_p->data<T>());
    ln_scale_p = ln_scale.get_ptr();
    ln_mean_p = ln_mean.get_ptr();
    ln_var_p = ln_var.get_ptr();
    d_ln_scale = ln_scale_grad;
    d_ln_bias = ln_bias_grad;
  } else {
    ln_scale_p = ln_scale_2.get_ptr();
    ln_mean_p = ln_mean_2.get_ptr();
    ln_var_p = ln_var_2.get_ptr();
    epsilon = ln_epsilon;
    d_ln_scale = ln_scale_2_grad;
    d_ln_bias = ln_bias_2_grad;
    bias_dropout_residual_out_ptr = reinterpret_cast<const XPUTypeT *>(
        bias_dropout_residual_out_p->data<T>());
  }

  ln_scale_ptr = ln_scale_p->data<float>();
  ln_mean_ptr = ln_mean_p->data<float>();
  ln_var_ptr = ln_var_p->data<float>();
  d_ln_scale_ptr = dev_ctx.template Alloc<float>(d_ln_scale);
  d_ln_bias_ptr = dev_ctx.template Alloc<float>(d_ln_bias);

  const auto input_x_dims = x.dims();
  const auto qkv_w_dims = qkv_weight.dims();

  int batch_size = input_x_dims[0];
  int seq_len = input_x_dims[1];
  int embed_dims = input_x_dims[2];
  num_heads = qkv_w_dims[1];
  int head_dims = qkv_w_dims[2];

  xpu::Context *xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  int r = 0;
  // int l3_total_size = xpu_ctx->_l3_mgr.get_size();
  XPUTypeT *d_ln_grad_ptr = NULL;       // dx5 [batch_size, seq_len, hidden]
  XPUTypeT *d_dropout_grad_ptr = NULL;  // dx5 [batch_size, seq_len, hidden]

  XPUTypeT *d_fmha_out_ptr =
      NULL;  // d_fmha_out [batch_size, seq_len, num_heads, head_dims]
  XPUTypeT *d_fmha_out_transpose_tmp_ptr =
      NULL;  // d_fmha_out_transpose [batch_size, seq_len, num_heads,
             // head_dims]

  XPUTypeT *d_qk_ptr =
      NULL;  // d_qk_ptr[batch_size, num_heads, seq_len, seq_len]

  XPUTypeT *d_combination_qkv_ptr =
      NULL;  // d_combination_qkv_ptr[3, batch_size, num_heads, seq_len,
             // head_dims]
  XPUTypeT *d_transpose_qkv_ptr =
      NULL;  // dx2 [batch_size, seq_len, 3, num_heads, head_dims]

  XPUTypeT *d_last_layernorm_grad_ptr =
      NULL;  // d_layer_out [batch_size, seq_len, embed_dims]

  const XPUTypeT *dy_input_ptr = d_y_ptr;

  d_ln_grad_ptr = RAII_GUARD.alloc<XPUTypeT>(batch_size * seq_len * embed_dims);
  d_dropout_grad_ptr =
      RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len * embed_dims);
  d_fmha_out_ptr = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len *
                                                       num_heads * head_dims);
  d_combination_qkv_ptr =
      RAII_GUARD.alloc<XPUTypeT>(batch_size * seq_len * embed_dims * 3);
  d_transpose_qkv_ptr = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(
      batch_size * seq_len * embed_dims * 3);
  d_fmha_out_transpose_tmp_ptr =
      RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len * embed_dims);
  d_qk_ptr = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len *
                                                 seq_len * num_heads);
  d_last_layernorm_grad_ptr =
      RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(batch_size * seq_len * embed_dims);

  if (pre_layer_norm == false) {
    r = xpu::layer_norm_grad(xpu_ctx,
                             bias_dropout_residual_out_ptr,
                             d_y_ptr,
                             d_ln_grad_ptr,
                             batch_size * seq_len,
                             embed_dims,
                             epsilon,
                             ln_scale_ptr,
                             ln_mean_ptr,
                             ln_var_ptr,
                             d_ln_scale_ptr,
                             d_ln_bias_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
    dy_input_ptr = d_ln_grad_ptr;
  }
  // dropout_grad
  DropoutGrad<XPUTypeT>(xpu_ctx,
                        dy_input_ptr,
                        dropout_mask_out_ptr,
                        d_dropout_grad_ptr,
                        dropout_param,
                        batch_size * num_heads * seq_len * head_dims);

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
  const XPUTypeT *a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
  const XPUTypeT *b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
  const XPUTypeT *a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
  const XPUTypeT *b_2 = reinterpret_cast<const XPUTypeT *>(NULL);

  XPUTypeT *c_1 = d_fmha_out_ptr;
  XPUTypeT *c_2 = d_out_linear_weight_ptr;
  phi::XpuFcInfo info_dfmha;
  phi::XpuFcInfo info_dlinear_w;

  std::tuple<phi::XpuFcInfo,
             phi::XpuFcInfo,
             const XPUTypeT *,
             const XPUTypeT *,
             const XPUTypeT *,
             const XPUTypeT *>
      fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                      &RAII_GUARD,
                                      linear_fc_info,
                                      false,
                                      false,
                                      fmha_out_ptr,
                                      out_linear_weight_ptr,
                                      d_dropout_grad_ptr);

  std::tie(info_dfmha, info_dlinear_w, a_1, b_1, a_2, b_2) = fc_info;
  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_2, b_2, c_2, info_dlinear_w, 1.0f, 0.f, true);

  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_1, b_1, c_1, info_dfmha, 1.0f, 0.f, true);

  // dlinear_bias
  r = xpu::reduce_sum(xpu_ctx,
                      d_dropout_grad_ptr,
                      d_out_linear_bias_ptr,
                      {batch_size * seq_len, embed_dims},
                      {0});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
  {
    int qkv_size = batch_size * seq_len * num_heads * head_dims;
    const XPUTypeT *q_out_ptr = qkv_transpose_out_ptr;
    const XPUTypeT *k_out_ptr = q_out_ptr + qkv_size;
    const XPUTypeT *v_out_ptr = k_out_ptr + qkv_size;
    XPUTypeT *d_q_out_ptr = d_combination_qkv_ptr;
    XPUTypeT *d_k_out_ptr = d_q_out_ptr + qkv_size;
    XPUTypeT *d_v_out_ptr = d_k_out_ptr + qkv_size;
    r = xpu::transpose<XPUTypeT>(xpu_ctx,
                                 d_fmha_out_ptr,
                                 d_fmha_out_transpose_tmp_ptr,
                                 {batch_size, seq_len, num_heads, head_dims},
                                 {0, 2, 1, 3});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

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

    const XPUTypeT *a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    const XPUTypeT *b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    const XPUTypeT *a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    const XPUTypeT *b_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    XPUTypeT *c_1 = d_qk_ptr;
    XPUTypeT *c_2 = d_v_out_ptr;
    phi::XpuFcInfo info_d_qk;
    phi::XpuFcInfo info_d_v;

    std::tuple<phi::XpuFcInfo,
               phi::XpuFcInfo,
               const XPUTypeT *,
               const XPUTypeT *,
               const XPUTypeT *,
               const XPUTypeT *>
        fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                        &RAII_GUARD,
                                        qktv_fc_info,
                                        false,
                                        false,
                                        attn_dropout_out_ptr,
                                        v_out_ptr,
                                        d_fmha_out_transpose_tmp_ptr);

    std::tie(info_d_qk, info_d_v, a_1, b_1, a_2, b_2) = fc_info;
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_1, b_1, c_1, info_d_qk, 1.0f, 0.f, true);
    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_2, b_2, c_2, info_d_v, 1.0f, 0.f, true);

    DropoutGrad<XPUTypeT>(xpu_ctx,
                          d_qk_ptr,
                          attn_dropout_mask_ptr,
                          d_qk_ptr,
                          attn_dropout_param,
                          batch_size * seq_len * seq_len * num_heads);

    r = xpu::softmax_grad<XPUTypeT>(xpu_ctx,
                                    softmax_out_ptr,
                                    d_qk_ptr,
                                    d_qk_ptr,
                                    {batch_size, num_heads, seq_len, seq_len},
                                    3);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax_grad");

    if (d_src_mask_out_ptr) {
      r = xpu::copy<XPUTypeT>(xpu_ctx,
                              d_qk_ptr,
                              d_src_mask_out_ptr,
                              batch_size * seq_len * seq_len * num_heads);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    }
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

    a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
    a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    b_2 = reinterpret_cast<const XPUTypeT *>(NULL);
    c_1 = d_q_out_ptr;
    c_2 = d_k_out_ptr;
    phi::XpuFcInfo info_d_q;
    phi::XpuFcInfo info_d_k;

    fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                    &RAII_GUARD,
                                    qk_fc_info,
                                    false,
                                    true,
                                    q_out_ptr,
                                    k_out_ptr,
                                    d_qk_ptr);

    std::tie(info_d_q, info_d_k, a_1, b_1, a_2, b_2) = fc_info;

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_1, b_1, c_1, info_d_q, 1.0f / sqrt(head_dims), 0.f, true);

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_2, b_2, c_2, info_d_k, 1.0f, 0.f, true);
  }

  //
  r = xpu::transpose<XPUTypeT>(xpu_ctx,
                               d_combination_qkv_ptr,
                               d_transpose_qkv_ptr,
                               {3, batch_size, num_heads, seq_len, head_dims},
                               {1, 3, 0, 2, 4});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  // dx and d_qkv_w
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

  a_1 = reinterpret_cast<const XPUTypeT *>(NULL);
  b_1 = reinterpret_cast<const XPUTypeT *>(NULL);
  a_2 = reinterpret_cast<const XPUTypeT *>(NULL);
  b_2 = reinterpret_cast<const XPUTypeT *>(NULL);
  c_1 = (pre_layer_norm == true) ? d_last_layernorm_grad_ptr : d_x_ptr;
  c_2 = d_qkv_weight_ptr;
  phi::XpuFcInfo info_d_x;
  phi::XpuFcInfo info_d_qkv_w;

  const XPUTypeT *use_calc_input_x_ptr =
      (pre_layer_norm == true) ? ln_out_ptr : input_x_ptr;

  fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                  &RAII_GUARD,
                                  qkv_fc_info,
                                  false,
                                  true,
                                  use_calc_input_x_ptr,
                                  qkv_weight_ptr,
                                  d_transpose_qkv_ptr);

  std::tie(info_d_x, info_d_qkv_w, a_1, b_1, a_2, b_2) = fc_info;
  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_1, b_1, c_1, info_d_x, 1.0f, 0.f, true);
  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_2, b_2, c_2, info_d_qkv_w, 1.0f, 0.f, true);

  // d_qkv_bias
  r = xpu::reduce_sum(xpu_ctx,
                      d_transpose_qkv_ptr,
                      d_qkv_bias_ptr,
                      {batch_size * seq_len, 3 * embed_dims},
                      {0});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

  if (pre_layer_norm) {
    r = xpu::layer_norm_grad(xpu_ctx,
                             input_x_ptr,
                             c_1,
                             d_x_ptr,
                             batch_size * seq_len,
                             embed_dims,
                             epsilon,
                             ln_scale_ptr,
                             ln_mean_ptr,
                             ln_var_ptr,
                             d_ln_scale_ptr,
                             d_ln_bias_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
  }

  // add rediaus dy
  r = xpu::add(xpu_ctx,
               dy_input_ptr,
               d_x_ptr,
               d_x_ptr,
               batch_size * seq_len * embed_dims);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
}
}  // namespace phi

PD_REGISTER_KERNEL(fused_attention_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::FusedAttentionGradKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(7).SetDataType(phi::DataType::FLOAT32);
}
