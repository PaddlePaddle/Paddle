/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/impl/matmul_grad_kernel_impl.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"
#include "paddle/phi/kernels/xpu/xpu_fused_common_function.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FFNGrad(const phi::XPUContext& dev_ctx,
             const phi::DenseTensor* d_out,
             const phi::DenseTensor* x,
             const phi::DenseTensor* dropout1_mask,
             const phi::DenseTensor* dropout2_mask,
             const phi::DenseTensor* linear1_out,
             const phi::DenseTensor* ln1_out,
             const phi::DenseTensor* dropout1_out,
             const phi::DenseTensor* dropout2_out,
             const phi::DenseTensor* linear1_weight,
             const phi::DenseTensor* linear2_weight,
             const phi::DenseTensor* ln_scale,
             const phi::DenseTensor* ln_mean,
             const phi::DenseTensor* ln_variance,
             phi::DenseTensor* d_x,
             phi::DenseTensor* d_linear1_weight,
             phi::DenseTensor* d_linear1_bias,
             phi::DenseTensor* d_linear2_weight,
             phi::DenseTensor* d_linear2_bias,
             phi::DenseTensor* d_ln_scale,
             phi::DenseTensor* d_ln_bias,
             const int bsz_seq,
             const int d_model,
             const int dim_feedforward,
             const XPUDropoutParam& dropout_param1,
             const XPUDropoutParam& dropout_param2,
             const std::string& act_method,
             const bool pre_layer_norm,
             const float epsilon,
             const int ring_id) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);
  int r = xpu::SUCCESS;

  // inputs ptr
  const XPUTypeT* d_out_ptr =
      reinterpret_cast<const XPUTypeT*>(d_out->data<T>());
  const XPUTypeT* x_ptr = reinterpret_cast<const XPUTypeT*>(x->data<T>());
  const XPUTypeT* dropout1_mask_ptr =
      reinterpret_cast<const XPUTypeT*>(dropout1_mask->data<T>());
  const XPUTypeT* dropout2_mask_ptr =
      reinterpret_cast<const XPUTypeT*>(dropout2_mask->data<T>());
  const XPUTypeT* linear1_out_ptr =
      reinterpret_cast<const XPUTypeT*>(linear1_out->data<T>());
  const XPUTypeT* dropout1_out_ptr =
      reinterpret_cast<const XPUTypeT*>(dropout1_out->data<T>());
  const XPUTypeT* linear1_weight_ptr =
      reinterpret_cast<const XPUTypeT*>(linear1_weight->data<T>());
  const XPUTypeT* linear2_weight_ptr =
      reinterpret_cast<const XPUTypeT*>(linear2_weight->data<T>());
  const float* ln_scale_ptr = ln_scale->data<float>();

  const float* ln_mean_ptr = ln_mean->data<float>();
  const float* ln_variance_ptr = ln_variance->data<float>();
  // outputs ptr
  XPUTypeT* d_x_ptr = reinterpret_cast<XPUTypeT*>(d_x->data<T>());
  XPUTypeT* d_linear1_weight_ptr =
      reinterpret_cast<XPUTypeT*>(d_linear1_weight->data<T>());
  XPUTypeT* d_linear1_bias_ptr =
      reinterpret_cast<XPUTypeT*>(d_linear1_bias->data<T>());
  XPUTypeT* d_linear2_weight_ptr =
      reinterpret_cast<XPUTypeT*>(d_linear2_weight->data<T>());
  XPUTypeT* d_linear2_bias_ptr =
      reinterpret_cast<XPUTypeT*>(d_linear2_bias->data<T>());
  float* d_ln_scale_ptr = d_ln_scale->data<float>();
  float* d_ln_bias_ptr = d_ln_bias->data<float>();

  size_t l3_total_size = xpu_ctx->_l3_mgr.get_size();

  XPUTypeT* big_tmp_l3_ptr = NULL;    // dim_feedforward * bsz_seq
  XPUTypeT* small_tmp_l3_ptr = NULL;  // d_model * bsz_seq
  XPUTypeT* big_tmp_gm_ptr = NULL;    // dim_feedforward * bsz_seq
  XPUTypeT* small_tmp_gm_ptr = NULL;  // d_model * bsz_seq

  XPUTypeT* d_layernorm_out_ptr = NULL;  // dx9
  XPUTypeT* d_dropout2_out_ptr = NULL;   // dx7

  XPUTypeT* d_linear2_out_ptr = NULL;   // dx5
  XPUTypeT* d_dropout1_out_ptr = NULL;  // dx4
  XPUTypeT* d_act_out_ptr = NULL;       // dx3

  XPUTypeT* d_linear1_out_ptr = NULL;  // dx1

  const XPUTypeT* d_residual_ptr = d_out_ptr;

  if (l3_total_size >=
      (dim_feedforward * bsz_seq * sizeof(T) + d_model * bsz_seq * sizeof(T))) {
    big_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_l3_ptr);
    small_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(d_model * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_l3_ptr);
    d_layernorm_out_ptr = small_tmp_l3_ptr;
    d_dropout2_out_ptr = small_tmp_l3_ptr;
    d_linear2_out_ptr = big_tmp_l3_ptr;
    d_dropout1_out_ptr = big_tmp_l3_ptr;
    d_act_out_ptr = big_tmp_l3_ptr;
    d_linear1_out_ptr = small_tmp_l3_ptr;
  } else if (l3_total_size >= dim_feedforward * bsz_seq * sizeof(T)) {
    big_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_l3_ptr);
    small_tmp_l3_ptr = big_tmp_l3_ptr;
    big_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_gm_ptr);
    small_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(d_model * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_gm_ptr);

    d_layernorm_out_ptr = small_tmp_l3_ptr;
    d_dropout2_out_ptr = small_tmp_gm_ptr;
    d_linear2_out_ptr = big_tmp_l3_ptr;
    d_dropout1_out_ptr = big_tmp_l3_ptr;
    d_act_out_ptr = big_tmp_gm_ptr;
    d_linear1_out_ptr = small_tmp_l3_ptr;

  } else if (l3_total_size >= d_model * bsz_seq * sizeof(T)) {
    big_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_gm_ptr);
    small_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(d_model * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_l3_ptr);

    d_layernorm_out_ptr = small_tmp_l3_ptr;
    d_dropout2_out_ptr = small_tmp_l3_ptr;
    d_linear2_out_ptr = big_tmp_gm_ptr;
    d_dropout1_out_ptr = big_tmp_gm_ptr;
    d_act_out_ptr = big_tmp_gm_ptr;
    d_linear1_out_ptr = small_tmp_l3_ptr;
  } else {
    big_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_gm_ptr);
    small_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(d_model * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_gm_ptr);
    d_layernorm_out_ptr = small_tmp_gm_ptr;
    d_dropout2_out_ptr = small_tmp_gm_ptr;
    d_linear2_out_ptr = big_tmp_gm_ptr;
    d_dropout1_out_ptr = big_tmp_gm_ptr;
    d_act_out_ptr = big_tmp_gm_ptr;
    d_linear1_out_ptr = small_tmp_gm_ptr;
  }

  if (pre_layer_norm == false) {
    const XPUTypeT* dropout2_out_ptr =
        reinterpret_cast<const XPUTypeT*>(dropout2_out->data<T>());
    r = xpu::layer_norm_grad(xpu_ctx,
                             dropout2_out_ptr,
                             d_out_ptr,
                             d_layernorm_out_ptr,
                             bsz_seq,
                             d_model,
                             epsilon,
                             ln_scale_ptr,
                             ln_mean_ptr,
                             ln_variance_ptr,
                             d_ln_scale_ptr,
                             d_ln_bias_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
    d_residual_ptr = d_layernorm_out_ptr;
  }
  phi::DropoutGrad(xpu_ctx,
                   d_residual_ptr,
                   dropout2_mask_ptr,
                   d_dropout2_out_ptr,
                   dropout_param2,
                   bsz_seq * d_model);
  // linear_grad2
  r = xpu::reduce_sum(
      xpu_ctx, d_dropout2_out_ptr, d_linear2_bias_ptr, {bsz_seq, d_model}, {0});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

  phi::XpuFcInfo linear2_fc_info;
  linear2_fc_info.InitFcInfo(0,
                             bsz_seq,
                             d_model,
                             dim_feedforward,
                             false,
                             false,
                             nullptr,
                             nullptr,
                             nullptr);

  const XPUTypeT* a_1 = reinterpret_cast<const XPUTypeT*>(NULL);
  const XPUTypeT* b_1 = reinterpret_cast<const XPUTypeT*>(NULL);
  const XPUTypeT* a_2 = reinterpret_cast<const XPUTypeT*>(NULL);
  const XPUTypeT* b_2 = reinterpret_cast<const XPUTypeT*>(NULL);
  XPUTypeT* c_1 = d_linear2_out_ptr;
  XPUTypeT* c_2 = d_linear2_weight_ptr;
  phi::XpuFcInfo info_d_dropout1;
  phi::XpuFcInfo info_dw2;

  std::tuple<phi::XpuFcInfo,
             phi::XpuFcInfo,
             const XPUTypeT*,
             const XPUTypeT*,
             const XPUTypeT*,
             const XPUTypeT*>
      fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                      &RAII_GUARD,
                                      linear2_fc_info,
                                      false,
                                      false,
                                      dropout1_out_ptr,
                                      linear2_weight_ptr,
                                      d_dropout2_out_ptr);

  std::tie(info_d_dropout1, info_dw2, a_1, b_1, a_2, b_2) = fc_info;

  // if l3_total_size >= dim_feedforward * bsz_seq * sizeof(T), first transpose
  if (l3_total_size >= dim_feedforward * bsz_seq * sizeof(T) &&
      info_dw2.trans_x) {
    r = xpu::transpose<XPUTypeT>(xpu_ctx,
                                 dropout1_out_ptr,
                                 big_tmp_l3_ptr,
                                 {bsz_seq, dim_feedforward},
                                 {1, 0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    a_2 = big_tmp_l3_ptr;
    info_dw2.trans_x = !info_dw2.trans_x;
    info_dw2.stride_x = info_dw2.k;
  }

  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_1, b_1, c_1, info_d_dropout1, 1.0f, 0.f, true);

  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_2, b_2, c_2, info_dw2, 1.0f, 0.f, true);

  // dropout_grad1
  DropoutGrad(xpu_ctx,
              d_linear2_out_ptr,
              dropout1_mask_ptr,
              d_dropout1_out_ptr,
              dropout_param1,
              bsz_seq * dim_feedforward);

  // act_grad
  if (act_method == "gelu") {
    r = xpu::gelu_grad(xpu_ctx,
                       linear1_out_ptr,
                       linear1_out_ptr,
                       d_dropout1_out_ptr,
                       d_act_out_ptr,
                       bsz_seq * dim_feedforward);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu_grad");
  } else if (act_method == "relu") {
    r = xpu::relu_grad(xpu_ctx,
                       linear1_out_ptr,
                       linear1_out_ptr,
                       d_dropout1_out_ptr,
                       d_act_out_ptr,
                       bsz_seq * dim_feedforward);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu_grad");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently only supports gelu or relu activation functions!"));
  }

  // linear1_grad
  r = xpu::reduce_sum(xpu_ctx,
                      d_act_out_ptr,
                      d_linear1_bias_ptr,
                      {bsz_seq, dim_feedforward},
                      {0});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

  phi::XpuFcInfo linear1_fc_info;
  linear1_fc_info.InitFcInfo(0,
                             bsz_seq,
                             dim_feedforward,
                             d_model,
                             false,
                             false,
                             nullptr,
                             nullptr,
                             nullptr);

  a_1 = reinterpret_cast<const XPUTypeT*>(NULL);
  b_1 = reinterpret_cast<const XPUTypeT*>(NULL);
  a_2 = reinterpret_cast<const XPUTypeT*>(NULL);
  b_2 = reinterpret_cast<const XPUTypeT*>(NULL);

  c_1 = (pre_layer_norm == true ? d_linear1_out_ptr : d_x_ptr);
  c_2 = d_linear1_weight_ptr;
  phi::XpuFcInfo info_dx;
  phi::XpuFcInfo info_dw1;

  const XPUTypeT* linear1_x_ptr =
      (pre_layer_norm == true
           ? reinterpret_cast<const XPUTypeT*>(ln1_out->data<T>())
           : x_ptr);

  if (l3_total_size >= d_model * bsz_seq * sizeof(T) && info_dw1.trans_x) {
    r = xpu::transpose<XPUTypeT>(
        xpu_ctx, linear1_x_ptr, small_tmp_l3_ptr, {bsz_seq, d_model}, {1, 0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    a_2 = small_tmp_l3_ptr;
    info_dw1.trans_x = !info_dw1.trans_x;
    info_dw1.stride_x = info_dw1.k;
  }

  fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                  &RAII_GUARD,
                                  linear1_fc_info,
                                  false,
                                  false,
                                  linear1_x_ptr,
                                  linear1_weight_ptr,
                                  d_act_out_ptr);

  std::tie(info_dx, info_dw1, a_1, b_1, a_2, b_2) = fc_info;

  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f, 0.f, true);

  phi::MatMulXPUFunction<XPUTypeT>(
      xpu_ctx, a_2, b_2, c_2, info_dw1, 1.0f, 0.f, true);

  if (pre_layer_norm) {
    r = xpu::layer_norm_grad(xpu_ctx,
                             x_ptr,
                             c_1,
                             c_1,
                             bsz_seq,
                             d_model,
                             epsilon,
                             ln_scale_ptr,
                             ln_mean_ptr,
                             ln_variance_ptr,
                             d_ln_scale_ptr,
                             d_ln_bias_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
  }

  r = xpu::add(xpu_ctx, c_1, d_residual_ptr, d_x_ptr, d_model * bsz_seq);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
}

template <typename T, typename Context>
void FusedFeedForwardGradKernel(
    const Context& dev_ctx,
    const DenseTensor& out_grad,
    const DenseTensor& x,
    const DenseTensor& linear1_weight,
    const paddle::optional<DenseTensor>& linear1_bias,
    const DenseTensor& linear2_weight,
    const DenseTensor& dropout1_mask,
    const DenseTensor& dropout2_mask,
    const DenseTensor& linear1_out,
    const DenseTensor& dropout1_out,
    const paddle::optional<DenseTensor>& dropout2_out,
    const paddle::optional<DenseTensor>& ln1_scale,
    const paddle::optional<DenseTensor>& ln1_bias,
    const paddle::optional<DenseTensor>& ln1_out,
    const paddle::optional<DenseTensor>& ln1_mean,
    const paddle::optional<DenseTensor>& ln1_variance,
    const paddle::optional<DenseTensor>& ln2_scale,
    const paddle::optional<DenseTensor>& ln2_bias,
    const paddle::optional<DenseTensor>& ln2_mean,
    const paddle::optional<DenseTensor>& ln2_variance,
    const paddle::optional<DenseTensor>& linear2_bias,
    bool pre_layer_norm,
    float ln1_epsilon,
    float ln2_epsilon,
    const std::string& act_method,
    float dropout1_prob,
    float dropout2_prob,
    const std::string& dropout1_implementation,
    const std::string& dropout2_implementation,
    bool is_test,
    bool dropout1_fix_seed,
    bool dropout2_fix_seed,
    int dropout1_seed_val,
    int dropout2_seed_val,
    bool add_residual,
    int ring_id,
    DenseTensor* x_grad,
    DenseTensor* linear1_weight_grad,
    DenseTensor* linear1_bias_grad,
    DenseTensor* linear2_weight_grad,
    DenseTensor* linear2_bias_grad,
    DenseTensor* ln1_scale_grad,
    DenseTensor* ln1_bias_grad,
    DenseTensor* ln2_scale_grad,
    DenseTensor* ln2_bias_grad) {
  // inputs
  auto* d_out = &out_grad;
  auto* x_ptr = &x;

  auto* dropout1_mask_ptr = &dropout1_mask;
  auto* dropout2_mask_ptr = &dropout2_mask;
  auto* linear1_out_ptr = &linear1_out;
  auto* ln1_out_ptr = pre_layer_norm ? ln1_out.get_ptr() : nullptr;

  auto* dropout1_out_ptr = &dropout1_out;
  auto* dropout2_out_ptr = dropout2_out.get_ptr();
  auto* linear1_weight_ptr = &linear1_weight;
  auto* linear2_weight_ptr = &linear2_weight;

  const phi::DenseTensor* ln_mean = nullptr;
  const phi::DenseTensor* ln_variance = nullptr;
  const phi::DenseTensor* ln_scale = nullptr;

  if (pre_layer_norm) {
    ln_mean = ln1_mean.get_ptr();
    ln_variance = ln1_variance.get_ptr();
    ln_scale = ln1_scale.get_ptr();
  } else {
    ln_mean = ln2_mean.get_ptr();
    ln_variance = ln2_variance.get_ptr();
    ln_scale = ln2_scale.get_ptr();
  }

  // output
  auto* d_x = x_grad;

  phi::DenseTensor* d_ln_scale = nullptr;
  phi::DenseTensor* d_ln_bias = nullptr;

  if (pre_layer_norm) {
    d_ln_scale = ln1_scale_grad;
    d_ln_bias = ln1_bias_grad;
  } else {
    d_ln_scale = ln2_scale_grad;
    d_ln_bias = ln2_bias_grad;
  }

  auto* d_linear1_weight = linear1_weight_grad;
  auto* d_linear1_bias = linear1_bias_grad;
  auto* d_linear2_weight = linear2_weight_grad;
  auto* d_linear2_bias = linear2_bias_grad;

  float epsilon = 0.0f;
  if (pre_layer_norm) {
    epsilon = ln1_epsilon;
  } else {
    epsilon = ln2_epsilon;
  }

  bool is_upscale_in_train_1 = dropout1_implementation == "upscale_in_train";
  bool is_upscale_in_train_2 = dropout2_implementation == "upscale_in_train";

  phi::XPUDropoutParam dropout_param1;
  dropout_param1.initXPUDropoutParam(dropout1_prob,
                                     is_upscale_in_train_1,
                                     is_test,
                                     dropout1_fix_seed,
                                     nullptr,
                                     dropout1_seed_val);
  phi::XPUDropoutParam dropout_param2;
  dropout_param2.initXPUDropoutParam(dropout2_prob,
                                     is_upscale_in_train_2,
                                     is_test,
                                     dropout2_fix_seed,
                                     nullptr,
                                     dropout2_seed_val);
  dev_ctx.template Alloc<T>(d_x);
  dev_ctx.template Alloc<float>(d_ln_scale);
  dev_ctx.template Alloc<float>(d_ln_bias);
  dev_ctx.template Alloc<T>(d_linear1_bias);
  dev_ctx.template Alloc<T>(d_linear2_bias);
  dev_ctx.template Alloc<T>(d_linear1_weight);
  dev_ctx.template Alloc<T>(d_linear2_weight);

  auto x_dim = x_ptr->dims();
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
      phi::RowMatrixFromVector(x_dim), 0, false);

  auto linear1_weight_dim = linear1_weight_ptr->dims();
  int d_model = linear1_weight_dim[0];
  int dim_feedforward = linear1_weight_dim[linear1_weight_dim.size() - 1];
  int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

  FFNGrad<T, Context>(dev_ctx,
                      d_out,
                      x_ptr,
                      dropout1_mask_ptr,
                      dropout2_mask_ptr,
                      linear1_out_ptr,
                      ln1_out_ptr,
                      dropout1_out_ptr,
                      dropout2_out_ptr,
                      linear1_weight_ptr,
                      linear2_weight_ptr,
                      ln_scale,
                      ln_mean,
                      ln_variance,
                      d_x,
                      d_linear1_weight,
                      d_linear1_bias,
                      d_linear2_weight,
                      d_linear2_bias,
                      d_ln_scale,
                      d_ln_bias,
                      bsz_seq,
                      d_model,
                      dim_feedforward,
                      dropout_param1,
                      dropout_param2,
                      act_method,
                      pre_layer_norm,
                      epsilon,
                      ring_id);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_feedforward_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedFeedForwardGradKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(7).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(8).SetDataType(phi::DataType::FLOAT32);
}
