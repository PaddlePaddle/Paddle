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

#include "paddle/phi/backends/gpu/gpu_context.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#endif
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

// Create the definition of Maximum
DEFINE_CUDA_ELEMENTWISE_OP(Maximum)
template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}
// Create the definition of Minimum
DEFINE_CUDA_ELEMENTWISE_OP(Minimum)
template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}
// Create the definition of Remainder
DEFINE_CUDA_ELEMENTWISE_OP(Remainder)
// Create the definition of FloorDivide
DEFINE_CUDA_ELEMENTWISE_OP(FloorDivide)
template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}
// Create the definition of Heaviside
DEFINE_CUDA_ELEMENTWISE_OP(ElementwiseHeaviside)
// Create the definition of Pow
DEFINE_CUDA_ELEMENTWISE_OP(ElementwisePow)
template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(maximum, KPS, ALL_LAYOUT, phi::MaximumKernel, float) {}
PD_REGISTER_KERNEL(maximum_raw, KPS, ALL_LAYOUT, phi::MaximumRawKernel, float) {
}
PD_REGISTER_KERNEL(minimum, KPS, ALL_LAYOUT, phi::MinimumKernel, float) {}
PD_REGISTER_KERNEL(minimum_raw, KPS, ALL_LAYOUT, phi::MinimumRawKernel, float) {
}
PD_REGISTER_KERNEL(floor_divide, KPS, ALL_LAYOUT, phi::FloorDivideKernel, int) {
}
PD_REGISTER_KERNEL(
    floor_divide_raw, KPS, ALL_LAYOUT, phi::FloorDivideRawKernel, int) {}
PD_REGISTER_KERNEL(
    elementwise_pow, KPS, ALL_LAYOUT, phi::ElementwisePowKernel, float) {}
PD_REGISTER_KERNEL(
    elementwise_pow_raw, KPS, ALL_LAYOUT, phi::ElementwisePowRawKernel, float) {
}

#else
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(
    fmax, KPS, ALL_LAYOUT, phi::FMaxKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    fmin, KPS, ALL_LAYOUT, phi::FMinKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(maximum_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(minimum_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(remainder_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::RemainderRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_heaviside_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::ElementwiseHeavisideRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
#endif
