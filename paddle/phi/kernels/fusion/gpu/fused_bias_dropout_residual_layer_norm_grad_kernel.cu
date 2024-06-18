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
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#endif

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusedBiasDropoutResidualLnGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& residual,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& ln_scale,
    const paddle::optional<DenseTensor>& ln_bias,
    const DenseTensor& ln_mean,
    const DenseTensor& ln_variance,
    const DenseTensor& bias_dropout_residual_out,
    const DenseTensor& dropout_mask_out,
    const DenseTensor& y_grad,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    DenseTensor* x_grad,
    DenseTensor* residual_grad,
    DenseTensor* bias_grad,
    DenseTensor* ln_scale_grad,
    DenseTensor* ln_bias_grad) {
  using U = LayerNormParamType<T>;
  auto* d_y_data = y_grad.data<T>();
  auto* ln_scale_data =
      (ln_scale.get_ptr() == nullptr ? nullptr : ln_scale->data<U>());
  auto* dropout_mask_out_data = dropout_mask_out.data<uint8_t>();
  auto* bias_dropout_residual_out_data = bias_dropout_residual_out.data<T>();
  auto* ln_mean_data = ln_mean.data<U>();
  auto* ln_var_data = ln_variance.data<U>();
  auto* d_x_data =
      dev_ctx.template Alloc<T>(x_grad, x_grad->numel() * sizeof(T));
  auto* d_residual_data = dev_ctx.template Alloc<T>(
      residual_grad, residual_grad->numel() * sizeof(T));
  DenseTensor bias_dropout_residual_out_grad;
  bias_dropout_residual_out_grad.Resize(bias_dropout_residual_out.dims());
  auto* d_bias_dropout_residual_out_data =
      dev_ctx.template Alloc<T>(&bias_dropout_residual_out_grad);
  auto* d_bias_data =
      (bias_grad == nullptr ? nullptr
                            : dev_ctx.template Alloc<T>(
                                  bias_grad, bias_grad->numel() * sizeof(T)));
  auto* d_ln_scale_data =
      (ln_scale_grad == nullptr
           ? nullptr
           : dev_ctx.template Alloc<U>(ln_scale_grad,
                                       ln_scale_grad->numel() * sizeof(U)));
  auto* d_ln_bias_data =
      (ln_bias_grad == nullptr
           ? nullptr
           : dev_ctx.template Alloc<U>(ln_bias_grad,
                                       ln_bias_grad->numel() * sizeof(U)));

  const auto input_x_dims = y_grad.dims();
  int bsz_seq = 1;
  for (int i = 0; i < input_x_dims.size() - 1; i++) {
    bsz_seq *= input_x_dims[i];
  }
  int dim_embed = input_x_dims[input_x_dims.size() - 1];
  phi::fusion::DropoutParam dropout_param(
      dropout_fix_seed,
      0,
      is_test,
      dropout_implementation == "upscale_in_train",
      dropout_rate,
      nullptr,
      dropout_seed);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      fused_dropout_layernorm_helper(
          dev_ctx, bsz_seq, dim_embed, dropout_param, ln_epsilon);
  fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
      dev_ctx,
      d_y_data,
      bias_dropout_residual_out_data,
      dropout_mask_out_data,
      ln_scale_data,
      ln_mean_data,
      ln_var_data,
      d_bias_dropout_residual_out_data,
      d_ln_scale_data,
      d_ln_bias_data,
      d_x_data,
      d_bias_data,
      d_residual_data);
}

}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(fused_bias_dropout_residual_layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasDropoutResidualLnGradKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(fused_bias_dropout_residual_layer_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasDropoutResidualLnGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
