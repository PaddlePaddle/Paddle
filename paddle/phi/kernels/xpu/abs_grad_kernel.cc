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

#include "paddle/phi/kernels/abs_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AbsGradKernel(const Context& ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   DenseTensor* dx) {
  ctx.template Alloc<T>(dx);
  int r = xpu::abs_grad(ctx.x_context(),
                        x.data<T>(),
                        dout.data<T>(),
                        dout.data<T>(),
                        dx->data<T>(),
                        x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "abs_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(abs_grad, XPU, ALL_LAYOUT, phi::AbsGradKernel, float) {}
