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

#include "paddle/phi/kernels/uniform_random_inplace_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UniformRandomInplaceGradKernel(const Context& ctx,
                                    const DenseTensor& out_grad,
                                    float min,
                                    float max,
                                    int seed,
                                    int diag_num,
                                    int diag_step,
                                    float diag_val,
                                    DenseTensor* x_grad) {
  if (x_grad) {
    auto* data = ctx.template Alloc<T>(x_grad);
    std::fill(data, data + x_grad->numel(), T(0));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_inplace_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomInplaceGradKernel,
                   float,
                   double) {}
