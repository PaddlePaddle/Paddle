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

#include "paddle/phi/kernels/huber_loss_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void HuberLossKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& label,
                     float delta,
                     DenseTensor* out,
                     DenseTensor* residual) {
  auto residual_data = dev_ctx.template Alloc<T>(residual);
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto in0_data = input.data<T>();
  auto in1_data = label.data<T>();

  int r = xpu::huber_loss<T>(dev_ctx.x_context(),
                             in0_data,
                             in1_data,
                             residual_data,
                             out_data,
                             input.numel(),
                             1,
                             delta);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "huber_loss");
}
}  // namespace phi

PD_REGISTER_KERNEL(huber_loss, XPU, ALL_LAYOUT, phi::HuberLossKernel, float) {}
