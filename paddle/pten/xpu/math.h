/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_XPU

#include "paddle/pten/core/base_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/xpu_header.h"

namespace pt {

using XPUDeviceContext = paddle::platform::XPUDeviceContext;

template <typename T>
void Sign(const XPUDeviceContext& dev_ctx,
          const BaseTensor& x,
          BaseTensor* out) {
  out->mutable_data<T>();
  auto xpu_context = dev_ctx.x_context();
  int r = xpu::activation_forward(xpu_context,
                                  xpu::Activation_t::SIGN,
                                  in.numel(),
                                  in.data<T>(),
                                  out->mutbale_data<T>());
  PADDLE_ENFORCE_EQ(r,
                    xpu::Error_t::SUCCESS,
                    platform::errors::Fatal("XPU sign kernel error!"));
}

}  // namespace pt

#endif
