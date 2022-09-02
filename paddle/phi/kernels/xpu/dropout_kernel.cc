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

#include "paddle/phi/kernels/dropout_kernel.h"

#include <memory>
#include <string>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto* y = out;
  const auto* x_data = x.data<T>();
  auto* y_data = dev_ctx.template Alloc<T>(y);
  float dropout_prob = p.to<float>();

  int is_upscale = (mode == "upscale_in_train");

  if (!is_test) {
    int seed_data = 0;
    if (seed_tensor.get_ptr() != nullptr) {
      if ((seed_tensor->place()).GetType() == phi::AllocationType::XPU) {
        paddle::memory::Copy(phi::CPUPlace(),
                             &seed_data,
                             seed_tensor->place(),
                             seed_tensor->data<int>(),
                             sizeof(int));
      } else {
        seed_data = *(seed_tensor->data<int>());
      }

    } else {
      seed_data = fix_seed ? seed : 0;
    }

    auto* mask_data = dev_ctx.template Alloc<T>(mask);
    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      int r = xpu::constant(dev_ctx.x_context(),
                            reinterpret_cast<XPUType*>(y_data),
                            y->numel(),
                            XPUType(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      r = xpu::constant(dev_ctx.x_context(),
                        reinterpret_cast<XPUType*>(mask_data),
                        mask->numel(),
                        XPUType(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      return;
    }
    int r = xpu::dropout(dev_ctx.x_context(),
                         reinterpret_cast<const XPUType*>(x.data<T>()),
                         reinterpret_cast<XPUType*>(y->data<T>()),
                         reinterpret_cast<XPUType*>(mask_data),
                         seed_data,
                         mask->numel(),
                         is_upscale,
                         dropout_prob);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout");
  } else {
    float scale =
        (is_upscale) ? (1.0) : (static_cast<float>(1.0f - dropout_prob));
    int r = xpu::scale(dev_ctx.x_context(),
                       reinterpret_cast<const XPUType*>(x_data),
                       reinterpret_cast<XPUType*>(y_data),
                       x.numel(),
                       false,
                       scale,
                       0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   XPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   phi::dtype::float16) {}
