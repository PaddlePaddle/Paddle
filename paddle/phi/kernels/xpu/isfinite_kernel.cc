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

#include "paddle/phi/kernels/isfinite_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void IsnanKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }
  auto* out_data = dev_ctx.template Alloc<bool>(out);
  int r = xpu::isnan<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(x.data<T>()),
                              out_data,
                              x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "isnan");
}

template <typename T, typename Context>
void IsfiniteKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }
  auto* out_data = dev_ctx.template Alloc<bool>(out);
  int r = xpu::isfinite<XPUType>(dev_ctx.x_context(),
                                 reinterpret_cast<const XPUType*>(x.data<T>()),
                                 out_data,
                                 x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "isfinite");
}

template <typename T, typename Context>
void IsinfKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<bool>(out);
    return;
  }
  auto* out_data = dev_ctx.template Alloc<bool>(out);
  int r = xpu::isinf<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(x.data<T>()),
                              out_data,
                              x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "isinf");
}

}  // namespace phi

PD_REGISTER_KERNEL(isnan,
                   XPU,
                   ALL_LAYOUT,
                   phi::IsnanKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_KERNEL(isfinite,
                   XPU,
                   ALL_LAYOUT,
                   phi::IsfiniteKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(isinf,
                   XPU,
                   ALL_LAYOUT,
                   phi::IsinfKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
