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

#include "paddle/phi/kernels/math_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/reduce.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

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

}  // namespace phi

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

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
