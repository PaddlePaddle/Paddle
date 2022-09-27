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

#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void GradAddXPUKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);
  auto x_shape = phi::vectorize<int>(x.dims());
  auto y_shape = phi::vectorize<int>(y.dims());
  int r = xpu::broadcast_add(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x.data<T>()),
                             reinterpret_cast<const XPUType*>(y.data<T>()),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             x_shape,
                             y_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
}

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  int axis,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  XPUElementwise<T, XPUType>(
      dev_ctx, x, y, axis, out, xpu::broadcast_add<XPUType>);
}

}  // namespace phi

PD_REGISTER_KERNEL(grad_add,
                   XPU,
                   ALL_LAYOUT,
                   phi::GradAddXPUKernel,
                   phi::dtype::float16,
                   float) {}
PD_REGISTER_KERNEL(
    add_raw, XPU, ALL_LAYOUT, phi::AddRawKernel, phi::dtype::float16, float) {}
