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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void AddCMulXPUKernel(const Context& ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& w,
                      DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto* x_data = x.data<T>();
  const auto* y_data = y.data<T>();
  const auto* w_data = w.data<T>();

  auto* out_data = ctx.template Alloc<T>(out);

#ifdef PADDLE_WITH_XPU_PLUGIN
  int r = xpu::plugin::fast_addcmul(ctx.x_context(),
                                    reinterpret_cast<const XPUType*>(w_data),
                                    reinterpret_cast<const XPUType*>(x_data),
                                    reinterpret_cast<const XPUType*>(y_data),
                                    reinterpret_cast<XPUType*>(out_data),
                                    x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fast_addcmul");
#else
  int r = xpu::addcmul(ctx.x_context(),
                       reinterpret_cast<const XPUType*>(w_data),
                       reinterpret_cast<const XPUType*>(x_data),
                       reinterpret_cast<const XPUType*>(y_data),
                       reinterpret_cast<XPUType*>(out_data),
                       1.0f,
                       x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "addcmul");
#endif
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(addcmul_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::AddCMulXPUKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
