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

#include "paddle/phi/kernels/rrelu_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& noise,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad) {
  const T* n_ptr = noise.data<T>();
  const T* x_ptr = x.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();
  int numel = static_cast<int>(x.numel());
  if (!x_grad) return;

  int i = 0;
  T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);
  for (i = 0; i < numel; i++) {
    x_grad_ptr[i] = x_ptr[i] > 0 ? out_grad_ptr[i] : n_ptr[i] * out_grad_ptr[i];
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    rrelu_grad, CPU, ALL_LAYOUT, phi::RReluGradKernel, float, double) {}
