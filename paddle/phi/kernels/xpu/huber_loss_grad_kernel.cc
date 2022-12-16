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

#include "paddle/phi/kernels/huber_loss_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void HuberLossGradKernel(const Context& dev_ctx,
                         const DenseTensor& residual,
                         const DenseTensor& out_grad,
                         float delta,
                         DenseTensor* input_grad,
                         DenseTensor* label_grad) {
  T* input_grad_data = nullptr;
  T* label_grad_data = nullptr;
  if (input_grad) {
    input_grad_data = dev_ctx.template Alloc<T>(input_grad);
  }
  if (label_grad) {
    label_grad_data = dev_ctx.template Alloc<T>(label_grad);
  }
  auto out_grad_data = out_grad.data<T>();
  auto residual_data = residual.data<T>();
  int r = xpu::huber_loss_grad<T>(dev_ctx.x_context(),
                                  residual_data,
                                  out_grad_data,
                                  input_grad_data,
                                  label_grad_data,
                                  out_grad.numel(),
                                  1,
                                  delta);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "huber_loss_grad");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    huber_loss_grad, XPU, ALL_LAYOUT, phi::HuberLossGradKernel, float) {}
