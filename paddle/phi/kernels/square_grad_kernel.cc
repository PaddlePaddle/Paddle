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

#include "paddle/phi/kernels/square_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T, typename Context>
void SquareGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      DenseTensor* x_grad) {
  MultiplyKernel<T>(dev_ctx,
                    out_grad,
                    Multiply<T>(dev_ctx, Full<T>(dev_ctx, {1}, 2.0), x),
                    x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(square_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SquareGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(square_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SquareGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif
