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

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusedBiasDropoutResidualLnKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& residual,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& ln_scale,
    const paddle::optional<DenseTensor>& ln_bias,
    const float dropout_rate,
    const bool is_test,
    const bool dropout_fix_seed,
    const int dropout_seed,
    const std::string& dropout_implementation,
    const float ln_epsilon,
    DenseTensor* y,
    DenseTensor* bias_dropout_residual_out,
    DenseTensor* dropout_mask_out,
    DenseTensor* ln_mean,
    DenseTensor* ln_variance) {
  using U = phi::funcs::LayerNormParamType<T>;
  auto* x_data = x.data<T>();
  auto* bias_data = (bias.get_ptr() == nullptr) ? nullptr : bias->data<T>();
  auto* residual_data = residual.data<T>();
  auto* ln_scale_data =
      (ln_scale.get_ptr() == nullptr ? nullptr : ln_scale->data<U>());
  auto* ln_bias_data =
      (ln_bias.get_ptr() == nullptr ? nullptr : ln_bias->data<U>());
  auto* bias_dropout_residual_out_data =
      dev_ctx.template Alloc<T>(bias_dropout_residual_out,
                                bias_dropout_residual_out->numel() * sizeof(T));
  auto* ln_mean_data =
      dev_ctx.template Alloc<U>(ln_mean, ln_mean->numel() * sizeof(U));
  auto* ln_var_data =
      dev_ctx.template Alloc<U>(ln_variance, ln_variance->numel() * sizeof(U));
  auto* dropout_mask_out_data =
      (dropout_mask_out == nullptr)
          ? nullptr
          : dev_ctx.template Alloc<uint8_t>(
                dropout_mask_out, dropout_mask_out->numel() * sizeof(uint8_t));
  auto* y_data = dev_ctx.template Alloc<T>(y, y->numel() * sizeof(T));

  const auto input_x_dims = x.dims();
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
  // output = layernorm(residual + dropout(input + bias))
  fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
      dev_ctx,
      x_data,
      residual_data,
      bias_data,
      ln_scale_data,
      ln_bias_data,
      bias_dropout_residual_out_data,
      dropout_mask_out_data,
      y_data,
      ln_mean_data,
      ln_var_data);
}
}  // namespace fusion
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(fused_bias_dropout_residual_layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasDropoutResidualLnKernel,
                   float,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
#else
PD_REGISTER_KERNEL(fused_bias_dropout_residual_layer_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasDropoutResidualLnKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
#endif
