// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/layer_norm_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename TW, typename Context>
void LayerNormKernelImpl(const Context& ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale,
                         const paddle::optional<DenseTensor>& bias,
                         float epsilon,
                         int begin_norm_axis,
                         DenseTensor* out,
                         DenseTensor* mean,
                         DenseTensor* variance) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeTW = typename XPUTypeTrait<TW>::Type;
  const auto& x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  int64_t left = matrix_dim[0];
  int64_t right = matrix_dim[1];

  const auto* x_data = x.data<T>();
  const auto* scale_data = scale.get_ptr() ? scale->data<TW>() : nullptr;
  const auto* bias_data = bias.get_ptr() ? bias->data<TW>() : nullptr;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  auto* out_data = ctx.template Alloc<T>(out);
  auto* mean_data = ctx.template Alloc<float>(mean);
  auto* variance_data = ctx.template Alloc<float>(variance);

  int r = xpu::layer_norm(ctx.x_context(),
                          reinterpret_cast<const XPUType*>(x_data),
                          reinterpret_cast<XPUType*>(out_data),
                          left,
                          right,
                          epsilon,
                          reinterpret_cast<const XPUTypeTW*>(scale_data),
                          reinterpret_cast<const XPUTypeTW*>(bias_data),
                          mean_data,
                          variance_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
}

template <typename T, typename Context>
void LayerNormKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int begin_norm_axis,
                     DenseTensor* out,
                     DenseTensor* mean,
                     DenseTensor* variance) {
  bool valid_scale = (scale.get_ptr() != nullptr);
  bool valid_bias = (bias.get_ptr() != nullptr);

  auto x_dtype = x.dtype();
  phi::DataType scale_bias_dtype;
  if (valid_scale) {
    scale_bias_dtype = scale->dtype();
    if (valid_bias) {
      PADDLE_ENFORCE_EQ(scale->dtype(),
                        bias->dtype(),
                        common::errors::InvalidArgument(
                            "This Scale and Bias of layer_norm op "
                            "should have the same data type."));
    }
  } else {
    scale_bias_dtype = valid_bias ? bias->dtype() : x_dtype;
  }

  bool is_scale_bias_same_dtype_with_x = (x_dtype == scale_bias_dtype);
  if (!is_scale_bias_same_dtype_with_x) {
    PADDLE_ENFORCE_EQ(scale_bias_dtype,
                      phi::DataType::FLOAT32,
                      common::errors::InvalidArgument(
                          "Unsupported data type of Scale and Bias"));
  }

  if (is_scale_bias_same_dtype_with_x) {
    LayerNormKernelImpl<T, T, Context>(
        ctx, x, scale, bias, epsilon, begin_norm_axis, out, mean, variance);
  } else {
    LayerNormKernelImpl<T, float, Context>(
        ctx, x, scale, bias, epsilon, begin_norm_axis, out, mean, variance);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(layer_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
