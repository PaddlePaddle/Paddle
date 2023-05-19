//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  AddRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  int axis = -1;
  SubtractRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace phi
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(subtract,
                   CPU,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(add,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}

#if defined(PADDLE_WITH_XPU_KP) && defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(subtract, KPS, ALL_LAYOUT, phi::SubtractKernel, float) {}
PD_REGISTER_KERNEL(add, KPS, ALL_LAYOUT, phi::AddKernel, float) {}
#elif defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(subtract,
                   KPS,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(add,
                   KPS,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
#endif

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)

PD_REGISTER_KERNEL(add,
                   XPU,
                   ALL_LAYOUT,
                   phi::AddKernel,
                   phi::dtype::float16,
                   float,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(subtract,
                   XPU,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   phi::dtype::float16,
                   int64_t) {}
#endif
