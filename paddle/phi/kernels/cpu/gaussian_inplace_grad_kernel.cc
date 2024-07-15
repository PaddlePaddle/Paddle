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

#include "paddle/phi/kernels/gaussian_inplace_grad_kernel.h"

#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// If T is not complex
template <
    typename T,
    typename Context,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianInplaceGrad(const Context& ctx, DenseTensor* x_grad) {
  if (x_grad) {
    auto* data = ctx.template Alloc<T>(x_grad);
    std::fill(data, data + x_grad->numel(), T(0));
  }
}

// If T is complex
template <
    typename T,
    typename Context,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void GaussianInplaceGrad(const Context& ctx, DenseTensor* x_grad) {
  if (x_grad) {
    auto* data = ctx.template Alloc<T>(x_grad);
    T value = T(static_cast<phi::dtype::Real<T>>(0.0f),
                static_cast<phi::dtype::Real<T>>(0.0f));
    std::fill(data, data + x_grad->numel(), value);
  }
}

template <typename T, typename Context>
void GaussianInplaceGradKernel(const Context& ctx,
                               const DenseTensor& out_grad UNUSED,
                               float mean UNUSED,
                               float std UNUSED,
                               int seed UNUSED,
                               DenseTensor* x_grad) {
  GaussianInplaceGrad<T>(ctx, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(gaussian_inplace_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::GaussianInplaceGradKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
