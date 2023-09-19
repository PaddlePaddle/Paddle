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

#include "paddle/phi/kernels/bce_loss_grad_kernel.h"

#include <algorithm>  // for max

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void BCELossGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& label,
                       const DenseTensor& out_grad,
                       DenseTensor* input_grad) {
  auto dx_data = dev_ctx.template Alloc<T>(input_grad);
  auto dout_data = out_grad.data<T>();
  auto x_data = input.data<T>();
  auto label_data = label.data<T>();

  int x_numel = static_cast<int>(input.numel());

  // dx = dout * ((x - label)/(x - x^2))
  for (int i = 0; i < x_numel; ++i) {
    dx_data[i] =
        dout_data[i] * ((x_data[i] - label_data[i]) /
                        std::max((static_cast<T>(1) - x_data[i]) * x_data[i],
                                 static_cast<T>(1e-12)));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    bce_loss_grad, CPU, ALL_LAYOUT, phi::BCELossGradKernel, float, double) {}
