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

#include "paddle/phi/kernels/square_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"

namespace phi {

template <typename T, typename Context>
void SquareKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  DenseTensor* out) {
  PowKernel<T>(dev_ctx, x, 2, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    square, CPU, ALL_LAYOUT, phi::SquareKernel, float, double, int, int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(square,
                   GPU,
                   ALL_LAYOUT,
                   phi::SquareKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
#endif
