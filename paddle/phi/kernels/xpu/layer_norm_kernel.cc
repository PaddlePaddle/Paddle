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

template <typename T, typename Context>
void LayerNormKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int begin_norm_axis,
                     bool is_test,
                     DenseTensor* out,
                     DenseTensor* mean,
                     DenseTensor* variance) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  const auto* x_data = x.data<T>();
  const auto* scale_data =
      (scale.get_ptr() == nullptr ? nullptr : scale.get_ptr()->data<float>());
  const auto* bias_data =
      (bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>());
  auto* out_data = ctx.template Alloc<T>(out);
  auto* mean_data = ctx.template Alloc<float>(mean);
  auto* variance_data = ctx.template Alloc<float>(variance);

  // int layer_norm(Context* ctx, const T* x, T* y, int m, int n, float eps,
  // const float* scale, const float* bias, float* mean, float* var);
  int r = xpu::layer_norm(ctx.x_context(),
                          reinterpret_cast<const XPUType*>(x_data),
                          reinterpret_cast<XPUType*>(out_data),
                          left,
                          right,
                          epsilon,
                          scale_data,
                          bias_data,
                          mean_data,
                          variance_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
}
}  // namespace phi

PD_REGISTER_KERNEL(layer_norm,
                   XPU,
                   ALL_LAYOUT,
                   phi::LayerNormKernel,
                   float,
                   phi::dtype::float16) {}
