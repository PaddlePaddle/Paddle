/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/softmax_kernel.h"

namespace phi {

template <typename T, typename Context>
void KLDivLossGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& label,
                         const DenseTensor& d_out,
                         const std::string& reduction,
                         bool log_target,
                         DenseTensor* d_x) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(d_x);
  if (d_x->numel() == 0) {
    return;
  }

  int r = XPU_SUCCESS;

  if (log_target) {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* label_exp = RAII_GUARD.alloc_l3_or_gm<XPUType>(label.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(label_exp);

    r = xpu::exp(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(label.data<T>()),
                 label_exp,
                 label.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "exp");

    r = xpu::kldiv_loss_grad(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(label_exp),
                             reinterpret_cast<const XPUType*>(d_out.data<T>()),
                             reinterpret_cast<XPUType*>(d_x->data<T>()),
                             d_x->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "kldiv_loss_grad");
  } else {
    r = xpu::kldiv_loss_grad(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(label.data<T>()),
                             reinterpret_cast<const XPUType*>(d_out.data<T>()),
                             reinterpret_cast<XPUType*>(d_x->data<T>()),
                             d_x->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "kldiv_loss_grad");
  }

  if ("none" != reduction) {
    PADDLE_THROW(common::errors::Unavailable(
        "Not supported reduction [%s] in kldiv_loss_grad", reduction));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    kldiv_loss_grad, XPU, ALL_LAYOUT, phi::KLDivLossGradKernel, float) {}
