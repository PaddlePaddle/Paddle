// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {

template <typename T, typename Context>
void LogLossXPUKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& label,
                      float epsilon_in,
                      DenseTensor* out) {
  auto* predict = &input;
  auto* labels = &label;
  auto* loss = out;
  auto epsilon = static_cast<T>(epsilon_in);
  dev_ctx.template Alloc<T>(loss);
  int n = predict->numel();
  int r = xpu::log_loss(dev_ctx.x_context(),
                        predict->data<T>(),
                        labels->data<T>(),
                        loss->data<T>(),
                        n,
                        epsilon);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "log_loss");
}
}  // namespace phi

PD_REGISTER_KERNEL(log_loss, XPU, ALL_LAYOUT, phi::LogLossXPUKernel, float) {}
