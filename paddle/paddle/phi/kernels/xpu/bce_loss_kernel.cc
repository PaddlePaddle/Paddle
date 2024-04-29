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

#include "paddle/phi/kernels/bce_loss_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BCELossKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& label,
                   DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);

  auto x_numel = input.numel();
  int r =
      xpu::bce_loss<XPUType>(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(input.data<T>()),
                             reinterpret_cast<const XPUType*>(label.data<T>()),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             x_numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "bce_loss");
}

}  // namespace phi

PD_REGISTER_KERNEL(bce_loss, XPU, ALL_LAYOUT, phi::BCELossKernel, float) {}
