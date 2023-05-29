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

#include "paddle/phi/kernels/dropout_grad_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DropoutGradRawKernel(const Context& dev_ctx,
                          const DenseTensor& mask,
                          const DenseTensor& out_grad,
                          const Scalar& p,
                          bool is_test,
                          const std::string& mode,
                          DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  PADDLE_ENFORCE_EQ(!is_test,
                    true,
                    phi::errors::InvalidArgument(
                        "GradOp is only callable when is_test is false"));
  auto* grad_x = x_grad;
  auto* grad_y = &out_grad;
  dev_ctx.template Alloc<T>(grad_x);
  float dropout_prob = p.to<float>();
  const T* mask_data = mask.data<T>();

  if (mode != "upscale_in_train") {
    int r = xpu::mul(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(grad_y->data<T>()),
                     reinterpret_cast<const XPUType*>(mask_data),
                     reinterpret_cast<XPUType*>(grad_x->data<T>()),
                     grad_y->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
    return;
  }

  auto version =
      phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
  if (version == phi::backends::xpu::XPUVersion::XPU1) {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    XPUType* mask_new = RAII_GUARD.alloc_l3_or_gm<XPUType>(mask.numel());
    float scale =
        (dropout_prob == 1.0f) ? (1.0f) : (1.0f / (1.0f - dropout_prob));
    int r = xpu::scale(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType*>(mask.data<T>()),
                       reinterpret_cast<XPUType*>(mask_new),
                       mask.numel(),
                       false,
                       scale,
                       0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    r = xpu::mul(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(grad_y->data<T>()),
                 reinterpret_cast<const XPUType*>(mask_new),
                 reinterpret_cast<XPUType*>(grad_x->data<T>()),
                 grad_y->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mul");
  } else {
    int r =
        xpu::dropout_grad(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(mask.data<T>()),
                          reinterpret_cast<const XPUType*>(grad_y->data<T>()),
                          reinterpret_cast<XPUType*>(grad_x->data<T>()),
                          dropout_prob,
                          grad_y->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_grad");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::DropoutGradRawKernel,
                   float,
                   phi::dtype::float16) {}
