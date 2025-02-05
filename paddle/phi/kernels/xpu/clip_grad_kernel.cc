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

#include "paddle/phi/kernels/clip_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ClipGradKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const Scalar& min,
                    const Scalar& max,
                    DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  int r =
      xpu::clamp_grad(ctx.x_context(),
                      reinterpret_cast<const XPUDataType*>(x.data<T>()),
                      reinterpret_cast<const XPUDataType*>(out_grad.data<T>()),
                      reinterpret_cast<XPUDataType*>(x_grad->data<T>()),
                      x.numel(),
                      static_cast<XPUDataType>(min.to<T>()),
                      static_cast<XPUDataType>(max.to<T>()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "clamp_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(clip_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int64_t,
                   int) {}
