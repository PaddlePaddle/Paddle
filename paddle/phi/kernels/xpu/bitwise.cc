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

#include "paddle/phi/kernels/bitwise_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BitwiseNotKernel(const Context& ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);
  int r = xpu::logical_not(ctx.x_context(),
                           reinterpret_cast<const XPUDataType*>(x.data<T>()),
                           reinterpret_cast<XPUDataType*>(out->data<T>()),
                           x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "logical_not");
}

template <typename T, typename Context>
void BitwiseAndKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out) {
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);
  int r = xpu::logical_and(ctx.x_context(),
                           reinterpret_cast<const XPUDataType*>(x.data<T>()),
                           reinterpret_cast<const XPUDataType*>(y.data<T>()),
                           reinterpret_cast<XPUDataType*>(out->data<T>()),
                           x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "logical_and");
}
}  // namespace phi

PD_REGISTER_KERNEL(bitwise_not, XPU, ALL_LAYOUT, phi::BitwiseNotKernel, bool) {}
PD_REGISTER_KERNEL(bitwise_and, XPU, ALL_LAYOUT, phi::BitwiseAndKernel, bool) {}
