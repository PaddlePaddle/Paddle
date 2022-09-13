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

#include <memory>

#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/memory/memcpy.h"

namespace phi {

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context& dev_ctx,
                                             const DenseTensor& x,
                                             const DenseTensor& label,
                                             const DenseTensor& out_grad,
                                             bool normalize,
                                             int ignore_index,
                                             DenseTensor* in_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  PADDLE_ENFORCE_EQ(x.place().GetType() == phi::AllocationType::XPU,
                    true,
                    errors::Unavailable("This kernel only runs on XPU."));

  dev_ctx.template Alloc<T>(in_grad);

  // allocate temp memory
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int* hit = RAII_GUARD.alloc_l3_or_gm<int>(x.numel());
  PADDLE_ENFORCE_NOT_NULL(
      hit, errors::External("XPU alloc_l3_or_gm returns nullptr"));

  int r = xpu::sigmoid_cross_entropy_with_logits_grad(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(x.data<T>()),
      reinterpret_cast<const XPUType*>(label.data<T>()),
      reinterpret_cast<const XPUType*>(out_grad.data<T>()),
      reinterpret_cast<XPUType*>(in_grad->data<T>()),
      1,
      x.numel(),
      hit,
      ignore_index);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sigmoid_cross_entropy_with_logits");
  if (normalize) {
    int* non_zero = RAII_GUARD.alloc_l3_or_gm<int>(1);
    PADDLE_ENFORCE_NOT_NULL(
        non_zero, errors::External("XPU alloc_l3_or_gm returns nullptr"));
    int r = xpu::nonzero_count(dev_ctx.x_context(),
                               reinterpret_cast<const XPUType*>(hit),
                               non_zero,
                               x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "nonzero_count");
    int non_zero_cpu = 0;
    paddle::memory::Copy(CPUPlace(),
                         static_cast<void*>(&non_zero_cpu),
                         dev_ctx.GetPlace(),
                         static_cast<void*>(non_zero),
                         sizeof(int));
    r = xpu::scale(dev_ctx.x_context(),
                   reinterpret_cast<const XPUType*>(in_grad->data<T>()),
                   reinterpret_cast<XPUType*>(in_grad->data<T>()),
                   x.numel(),
                   false,
                   1.0f / static_cast<float>(non_zero_cpu),
                   0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsGradKernel,
                   float) {}
