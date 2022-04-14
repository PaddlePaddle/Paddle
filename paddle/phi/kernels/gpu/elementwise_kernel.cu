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
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                             \
  template <typename T, typename Context>                            \
  void name##RawKernel(const Context& dev_ctx,                       \
                       const DenseTensor& x,                         \
                       const DenseTensor& y,                         \
                       int axis,                                     \
                       DenseTensor* out) {                           \
    std::vector<const DenseTensor*> inputs;                          \
    std::vector<DenseTensor*> outputs;                               \
    inputs.emplace_back(&x);                                         \
    inputs.emplace_back(&y);                                         \
    outputs.emplace_back(out);                                       \
    dev_ctx.template Alloc<T>(out);                                  \
    funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(          \
        dev_ctx, inputs, &outputs, axis, funcs::name##Functor<T>()); \
  }

/**
 * Kernels
 */

// Create the definition of Add
DEFINE_CUDA_ELEMENTWISE_OP(Add)
// Create the definition of Subtract
DEFINE_CUDA_ELEMENTWISE_OP(Subtract)
// Create the definition of Multiply
DEFINE_CUDA_ELEMENTWISE_OP(Multiply)
// Create the definition of Divide
DEFINE_CUDA_ELEMENTWISE_OP(Divide)
// Create the definition of Maximum
DEFINE_CUDA_ELEMENTWISE_OP(Maximum)
// Create the definition of Minimum
DEFINE_CUDA_ELEMENTWISE_OP(Minimum)
// Create the definition of Modulo
DEFINE_CUDA_ELEMENTWISE_OP(Modulo)
// Create the definition of FloorDivide
DEFINE_CUDA_ELEMENTWISE_OP(FloorDivide)
// Create the definition of Pow
DEFINE_CUDA_ELEMENTWISE_OP(ElementwisePow)

}  // namespace phi

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

PD_REGISTER_KERNEL(
    fmax, GPU, ALL_LAYOUT, phi::FMaxKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    fmin, GPU, ALL_LAYOUT, phi::FMinKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(add_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::AddRawKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(subtract_raw,
                   GPU,
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
PD_REGISTER_KERNEL(divide_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::DivideRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(multiply_raw,
                   GPU,
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
PD_REGISTER_KERNEL(maximum_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(minimum_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(modulo_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::ModuloRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
