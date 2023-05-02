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

#include "paddle/phi/kernels/gelu_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    bool approximate,
                    DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(x_grad);
  int r = xpu::gelu_grad<XPUType>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      nullptr,
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(x_grad->data<T>()),
      x_grad->numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(gelu_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::GeluGradKernel,
                   float,
                   phi::dtype::float16) {}
