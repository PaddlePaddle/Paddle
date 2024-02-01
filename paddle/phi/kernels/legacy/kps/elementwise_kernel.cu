// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

DEFINE_CUDA_ELEMENTWISE_OP(Add)
DEFINE_CUDA_ELEMENTWISE_OP(Divide)
DEFINE_CUDA_ELEMENTWISE_OP(Multiply)
DEFINE_CUDA_ELEMENTWISE_OP(Subtract)
DEFINE_CUDA_ELEMENTWISE_OP(Maximum)
DEFINE_CUDA_ELEMENTWISE_OP(Minimum)
DEFINE_CUDA_ELEMENTWISE_OP(Remainder)
DEFINE_CUDA_ELEMENTWISE_OP(FloorDivide)
DEFINE_CUDA_ELEMENTWISE_OP(ElementwisePow)

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(add_raw, KPS, ALL_LAYOUT, phi::AddRawKernel, float) {}
PD_REGISTER_KERNEL(divide_raw, KPS, ALL_LAYOUT, phi::DivideRawKernel, float) {}
PD_REGISTER_KERNEL(
    multiply_raw, KPS, ALL_LAYOUT, phi::MultiplyRawKernel, float) {}
PD_REGISTER_KERNEL(
    subtract_raw, KPS, ALL_LAYOUT, phi::SubtractRawKernel, float) {}
PD_REGISTER_KERNEL(maximum_raw, KPS, ALL_LAYOUT, phi::MaximumRawKernel, float) {
}
PD_REGISTER_KERNEL(minimum_raw, KPS, ALL_LAYOUT, phi::MinimumRawKernel, float) {
}
PD_REGISTER_KERNEL(
    floor_divide_raw, KPS, ALL_LAYOUT, phi::FloorDivideRawKernel, int) {}
PD_REGISTER_KERNEL(
    elementwise_pow_raw, KPS, ALL_LAYOUT, phi::ElementwisePowRawKernel, float) {
}
#else

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(add_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::AddRawKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(divide_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::DivideRawKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(multiply_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MultiplyRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   float16,
                   complex64,
                   complex128,
                   bfloat16) {}

PD_REGISTER_KERNEL(subtract_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::SubtractRawKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

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
                   float16,
                   int64_t,
                   bfloat16) {}
PD_REGISTER_KERNEL(floor_divide_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   double,
                   int,
                   float16,
                   int64_t,
                   bfloat16) {}
#endif
