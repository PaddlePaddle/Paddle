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

#include "paddle/phi/kernels/layer_norm_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LayerNormGradKernel(const Context& ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& scale,
                         const paddle::optional<DenseTensor>& bias,
                         const DenseTensor& mean,
                         const DenseTensor& variance,
                         const DenseTensor& out_grad,
                         float epsilon,
                         int begin_norm_axis,
                         bool is_test,
                         DenseTensor* x_grad,
                         DenseTensor* scale_grad,
                         DenseTensor* bias_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  const auto* x_data = x.data<T>();
  const auto* out_grad_data = out_grad.data<T>();
  const auto* mean_data = mean.data<float>();
  const auto* variance_data = variance.data<float>();
  const auto* scale_data =
      (scale.get_ptr() == nullptr ? nullptr : scale.get_ptr()->data<float>());
  auto* scale_grad_data =
      (scale_grad == nullptr ? nullptr : ctx.template Alloc<float>(scale_grad));
  auto* bias_grad_data =
      (bias_grad == nullptr ? nullptr : ctx.template Alloc<float>(bias_grad));
  auto* x_grad_data =
      (x_grad == nullptr ? nullptr : ctx.template Alloc<T>(x_grad));

  // int layer_norm_grad(Context* ctx, const T* x, const T* dy, T* dx, int m,
  // int n, float eps, const float* scale, const float* mean, const float*
  // var, float* dscale, float* dbias);
  int r = xpu::layer_norm_grad(ctx.x_context(),
                               reinterpret_cast<const XPUType*>(x_data),
                               reinterpret_cast<const XPUType*>(out_grad_data),
                               reinterpret_cast<XPUType*>(x_grad_data),
                               left,
                               right,
                               epsilon,
                               scale_data,
                               mean_data,
                               variance_data,
                               scale_grad_data,
                               bias_grad_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(layer_norm_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::LayerNormGradKernel,
                   float,
                   phi::dtype::float16) {}
