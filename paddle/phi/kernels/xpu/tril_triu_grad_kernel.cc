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

#include "paddle/phi/kernels/tril_triu_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TrilTriuGradKernel(const Context& ctx,
                        const DenseTensor& out_grad,
                        int diagonal,
                        bool lower,
                        DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(x_grad);
  auto dy_shape = vectorize<int>(out_grad.dims());
  int r = 0;
  if (lower) {
    r = xpu::tril(ctx.x_context(),
                  reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                  reinterpret_cast<XPUType*>(x_grad->data<T>()),
                  dy_shape,
                  diagonal);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "tril_op");
  } else {
    r = xpu::triu(ctx.x_context(),
                  reinterpret_cast<const XPUType*>(out_grad.data<T>()),
                  reinterpret_cast<XPUType*>(x_grad->data<T>()),
                  dy_shape,
                  diagonal);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "triu_op");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    tril_triu_grad, XPU, ALL_LAYOUT, phi::TrilTriuGradKernel, int, float) {}
