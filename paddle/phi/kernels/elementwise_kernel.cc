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
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  int axis = -1;
  RemainderRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void ElementwiseHeavisideKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& y,
                                DenseTensor* out) {
  int axis = -1;
  ElementwiseHeavisideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
  DivideRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  MultiplyRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

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

PD_REGISTER_KERNEL(maximum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder,
                   CPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(
    floor_divide, CPU, ALL_LAYOUT, phi::FloorDivideKernel, int, int64_t) {}
PD_REGISTER_KERNEL(elementwise_heaviside,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwiseHeavisideKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

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

PD_REGISTER_KERNEL(multiply,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(divide,
                   CPU,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

PD_REGISTER_KERNEL(maximum,
                   KPS,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum,
                   KPS,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder,
                   GPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    floor_divide, KPS, ALL_LAYOUT, phi::FloorDivideKernel, int, int64_t) {}
PD_REGISTER_KERNEL(elementwise_heaviside,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwiseHeavisideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   KPS,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}

#endif

#if defined(PADDLE_WITH_XPU_KP) && !defined(PADDLE_WITH_XPU)
PD_REGISTER_KERNEL(subtract, KPS, ALL_LAYOUT, phi::SubtractKernel, float) {}
PD_REGISTER_KERNEL(add, KPS, ALL_LAYOUT, phi::AddKernel, float) {}
PD_REGISTER_KERNEL(multiply, KPS, ALL_LAYOUT, phi::MultiplyKernel, float) {}
PD_REGISTER_KERNEL(divide, KPS, ALL_LAYOUT, phi::DivideKernel, float) {}
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

PD_REGISTER_KERNEL(multiply,
                   KPS,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(divide,
                   KPS,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   complex64,
                   complex128) {}
#endif

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)

PD_REGISTER_KERNEL(
    divide, XPU, ALL_LAYOUT, phi::DivideKernel, phi::dtype::float16, float) {}

PD_REGISTER_KERNEL(
    add, XPU, ALL_LAYOUT, phi::AddKernel, phi::dtype::float16, float) {}

PD_REGISTER_KERNEL(multiply,
                   XPU,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   phi::dtype::float16,
                   float) {}
PD_REGISTER_KERNEL(subtract,
                   XPU,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   phi::dtype::float16) {}
#endif

#if defined PADDLE_WITH_XPU
PD_REGISTER_KERNEL(floor_divide,
                   XPU,
                   ALL_LAYOUT,
                   phi::FloorDivideKernel,
                   float,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    maximum, XPU, ALL_LAYOUT, phi::MaximumKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(
    minimum, XPU, ALL_LAYOUT, phi::MinimumKernel, float, phi::dtype::float16) {}
PD_REGISTER_KERNEL(remainder,
                   XPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   phi::dtype::float16,
                   int32_t,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   XPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   phi::dtype::float16) {}
#endif
