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
void FFN(const phi::XPUContext& dev_ctx,
         const phi::DenseTensor* x,
         const phi::DenseTensor* linear1_weight,
         const phi::DenseTensor* linear1_bias,
         const phi::DenseTensor* linear2_weight,
         const phi::DenseTensor* linear2_bias,
         const phi::DenseTensor* ln_scale,
         const phi::DenseTensor* ln_bias,
         phi::DenseTensor* out,
         phi::DenseTensor* dropout1_mask,
         phi::DenseTensor* dropout2_mask,
         phi::DenseTensor* ln_mean,
         phi::DenseTensor* ln_variance,
         phi::DenseTensor* linear1_out,
         phi::DenseTensor* ln1_out,
         phi::DenseTensor* dropout1_out,
         phi::DenseTensor* dropout2_out,
         const int bsz_seq,
         const int d_model,
         const int dim_feedforward,
         const std::string& act_method,
         const bool pre_layer_norm,
         const float epsilon1,
         const float epsilon2,
         const phi::XPUDropoutParam& dropout_param1,
         const phi::XPUDropoutParam& dropout_param2,
         int ring_id) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;
  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  int r = xpu::SUCCESS;

  const XPUTypeT* x_ptr = reinterpret_cast<const XPUTypeT*>(x->data<T>());
  const XPUTypeT* residual_ptr = x_ptr;
  const XPUTypeT* linear1_weight_ptr =
      reinterpret_cast<const XPUTypeT*>(linear1_weight->data<T>());
  const XPUTypeT* linear1_bias_ptr =
      reinterpret_cast<const XPUTypeT*>(linear1_bias->data<T>());
  const XPUTypeT* linear2_weight_ptr =
      reinterpret_cast<const XPUTypeT*>(linear2_weight->data<T>());
  const XPUTypeT* linear2_bias_ptr =
      reinterpret_cast<const XPUTypeT*>(linear2_bias->data<T>());

  const float* ln_scale_ptr = ln_scale->data<float>();

  const float* ln_bias_ptr = ln_bias->data<float>();

  // out
  XPUTypeT* out_ptr = reinterpret_cast<XPUTypeT*>(out->data<T>());
  XPUTypeT* linear1_out_ptr =
      reinterpret_cast<XPUTypeT*>(linear1_out->data<T>());
  XPUTypeT* dropout1_mask_ptr =
      reinterpret_cast<XPUTypeT*>(dropout1_mask->data<T>());
  XPUTypeT* dropout2_mask_ptr =
      reinterpret_cast<XPUTypeT*>(dropout2_mask->data<T>());
  float* ln_mean_ptr = ln_mean->data<float>();
  float* ln_variance_ptr = ln_variance->data<float>();

  XPUTypeT* dropout1_out_ptr =
      reinterpret_cast<XPUTypeT*>(dropout1_out->data<T>());
  XPUTypeT* dropout2_out_ptr =
      reinterpret_cast<XPUTypeT*>(dropout2_out->data<T>());

  size_t l3_total_size = xpu_ctx->_l3_mgr.get_size();
  XPUTypeT* linear2_before_tmp_ptr = NULL;  // dim_feedforward * bsz_seq
  XPUTypeT* linear2_after_tmp_ptr = NULL;   // d_model * bsz_seq
  if (l3_total_size >= dim_feedforward * bsz_seq * sizeof(T)) {
    XPUTypeT* l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(l3_ptr);
    linear2_before_tmp_ptr = linear2_after_tmp_ptr = l3_ptr;
  } else if ((l3_total_size < dim_feedforward * bsz_seq * sizeof(T)) &&
             (l3_total_size >= d_model * bsz_seq * sizeof(T))) {
    XPUTypeT* l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(d_model * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(l3_ptr);
    linear2_after_tmp_ptr = l3_ptr;
    linear2_before_tmp_ptr =
        RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(linear2_before_tmp_ptr);

  } else {
    XPUTypeT* gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
    PADDLE_ENFORCE_XDNN_NOT_NULL(gm_ptr);
    linear2_before_tmp_ptr = linear2_after_tmp_ptr = gm_ptr;
  }

  // layernorm
  if (pre_layer_norm) {
    XPUTypeT* ln1_out_ptr = reinterpret_cast<XPUTypeT*>(ln1_out->data<T>());
    r = xpu::layer_norm(xpu_ctx,
                        x_ptr,
                        ln1_out_ptr,
                        bsz_seq,
                        d_model,
                        epsilon1,
                        ln_scale_ptr,
                        ln_bias_ptr,
                        ln_mean_ptr,
                        ln_variance_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm ");
    x_ptr = ln1_out_ptr;
  }

  // fc
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
  phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                   x_ptr,
                                   linear1_weight_ptr,
                                   linear2_before_tmp_ptr,
                                   linear1_fc_info,
                                   1.0f);

  // bias
  r = xpu::broadcast_add(xpu_ctx,
                         linear2_before_tmp_ptr,
                         linear1_bias_ptr,
                         linear1_out_ptr,
                         {bsz_seq, dim_feedforward},
                         {dim_feedforward});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  // act
  if (act_method == "gelu") {
    r = xpu::gelu(
        xpu_ctx, linear1_out_ptr, linear2_before_tmp_ptr, linear1_out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu");
  } else if (act_method == "relu") {
    r = xpu::relu(
        xpu_ctx, linear1_out_ptr, linear2_before_tmp_ptr, linear1_out->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently only supports gelu or relu activation functions!"));
  }

  // dropout1
  phi::Dropout<XPUTypeT>(xpu_ctx,
                         linear2_before_tmp_ptr,
                         dropout1_mask_ptr,
                         dropout1_out_ptr,
                         dropout_param1,
                         dropout1_out->numel());

  // fc
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
  phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                   dropout1_out_ptr,
                                   linear2_weight_ptr,
                                   dropout2_out_ptr,
                                   linear2_fc_info,
                                   1.0f);

  // bias
  r = xpu::broadcast_add(xpu_ctx,
                         dropout2_out_ptr,
                         linear2_bias_ptr,
                         dropout2_out_ptr,
                         {bsz_seq, d_model},
                         {d_model});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  // dropout2
  phi::Dropout<XPUTypeT>(xpu_ctx,
                         dropout2_out_ptr,
                         dropout2_mask_ptr,
                         dropout2_out_ptr,
                         dropout_param2,
                         dropout2_out->numel());

  // residual_ptr + dropout_out
  XPUTypeT* residual_add_out_ptr = out_ptr;
  if (pre_layer_norm == false) {
    residual_add_out_ptr = dropout2_out_ptr;
  }
  r = xpu::broadcast_add(xpu_ctx,
                         residual_ptr,
                         dropout2_out_ptr,
                         residual_add_out_ptr,
                         {bsz_seq, d_model},
                         {bsz_seq, d_model});
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

  if (pre_layer_norm == false) {
    r = xpu::layer_norm(xpu_ctx,
                        residual_add_out_ptr,
                        out_ptr,
                        bsz_seq,
                        d_model,
                        epsilon2,
                        ln_scale_ptr,
                        ln_bias_ptr,
                        ln_mean_ptr,
                        ln_variance_ptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
  }
}

template <typename T, typename Context>
void FusedFeedForwardKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& dropout1_seed,
                            const paddle::optional<DenseTensor>& dropout2_seed,
                            const DenseTensor& linear1_weight,
                            const paddle::optional<DenseTensor>& linear1_bias,
                            const DenseTensor& linear2_weight,
                            const paddle::optional<DenseTensor>& linear2_bias,
                            const paddle::optional<DenseTensor>& ln1_scale,
                            const paddle::optional<DenseTensor>& ln1_bias,
                            const paddle::optional<DenseTensor>& ln2_scale,
                            const paddle::optional<DenseTensor>& ln2_bias,
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
                            DenseTensor* out,
                            DenseTensor* dropout1_mask,
                            DenseTensor* dropout2_mask,
                            DenseTensor* ln1_mean,
                            DenseTensor* ln1_variance,
                            DenseTensor* ln2_mean,
                            DenseTensor* ln2_variance,
                            DenseTensor* linear1_out,
                            DenseTensor* ln1_out,
                            DenseTensor* dropout1_out,
                            DenseTensor* dropout2_out) {
  auto* x_ptr = &x;
  auto* linear1_weight_ptr = &linear1_weight;
  auto* linear1_bias_ptr = linear1_bias.get_ptr();
  auto* linear2_weight_ptr = &linear2_weight;
  auto* linear2_bias_ptr = linear2_bias.get_ptr();

  const phi::DenseTensor* ln_scale = nullptr;
  const phi::DenseTensor* ln_bias = nullptr;
  phi::DenseTensor* ln_mean = nullptr;
  phi::DenseTensor* ln_variance = nullptr;

  if (pre_layer_norm) {
    ln_scale = ln1_scale.get_ptr();
    ln_bias = ln1_bias.get_ptr();
    ln_mean = ln1_mean;
    ln_variance = ln1_variance;
    dev_ctx.template Alloc<T>(ln1_out);
  } else {
    ln_scale = ln2_scale.get_ptr();
    ln_bias = ln2_bias.get_ptr();
    ln_mean = ln2_mean;
    ln_variance = ln2_variance;
  }

  const float epsilon1 = ln1_epsilon;
  const float epsilon2 = ln2_epsilon;

  bool is_upscale_in_train_1 = dropout1_implementation == "upscale_in_train";
  bool is_upscale_in_train_2 = dropout2_implementation == "upscale_in_train";

  auto* dropout1_seed_ptr = dropout1_seed.get_ptr();
  auto* dropout2_seed_ptr = dropout2_seed.get_ptr();
  phi::XPUDropoutParam dropout_param1;
  dropout_param1.initXPUDropoutParam(dropout1_prob,
                                     is_upscale_in_train_1,
                                     is_test,
                                     dropout1_fix_seed,
                                     dropout1_seed_ptr,
                                     dropout1_seed_val);
  phi::XPUDropoutParam dropout_param2;
  dropout_param2.initXPUDropoutParam(dropout2_prob,
                                     is_upscale_in_train_2,
                                     is_test,
                                     dropout2_fix_seed,
                                     dropout2_seed_ptr,
                                     dropout2_seed_val);

  dev_ctx.template Alloc<float>(ln_mean);
  dev_ctx.template Alloc<float>(ln_variance);

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(dropout1_mask);
  dev_ctx.template Alloc<T>(dropout2_mask);
  dev_ctx.template Alloc<T>(dropout1_out);
  dev_ctx.template Alloc<T>(dropout2_out);
  dev_ctx.template Alloc<T>(linear1_out);

  auto x_dim = x_ptr->dims();
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
      phi::RowMatrixFromVector(x_dim), 0, false);

  auto dim = linear1_weight_ptr->dims();
  int d_model = dim[0];
  int dim_feedforward = dim[dim.size() - 1];
  int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

  phi::fusion::FFN<T, Context>(dev_ctx,
                               x_ptr,
                               linear1_weight_ptr,
                               linear1_bias_ptr,
                               linear2_weight_ptr,
                               linear2_bias_ptr,
                               ln_scale,
                               ln_bias,
                               out,
                               dropout1_mask,
                               dropout2_mask,
                               ln_mean,
                               ln_variance,
                               linear1_out,
                               ln1_out,
                               dropout1_out,
                               dropout2_out,
                               bsz_seq,
                               d_model,
                               dim_feedforward,
                               act_method,
                               pre_layer_norm,
                               epsilon1,
                               epsilon2,
                               dropout_param1,
                               dropout_param2,
                               ring_id);
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_feedforward,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedFeedForwardKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
}
