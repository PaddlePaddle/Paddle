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

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/kernels/elementwise_kernel.h"
#include "paddle/pten/kernels/funcs/elementwise_functor.h"
#include "paddle/pten/kernels/gpu/elementwise_impl.cu.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                             \
  template <typename T, typename ContextT>                           \
  void name(const ContextT& dev_ctx,                                 \
            const DenseTensor& x,                                    \
            const DenseTensor& y,                                    \
            int axis,                                                \
            DenseTensor* out) {                                      \
    std::vector<const DenseTensor*> inputs;                          \
    std::vector<DenseTensor*> outputs;                               \
    inputs.emplace_back(&x);                                         \
    inputs.emplace_back(&y);                                         \
    outputs.emplace_back(out);                                       \
    out->mutable_data<T>();                                          \
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(     \
        dev_ctx, inputs, &outputs, axis, funcs::name##Functor<T>()); \
  }

// Create the definition of Add
DEFINE_CUDA_ELEMENTWISE_OP(Add)
// Create the definition of Subtract
DEFINE_CUDA_ELEMENTWISE_OP(Subtract)
// Create the definition of Multiply
DEFINE_CUDA_ELEMENTWISE_OP(Multiply)
// Create the definition of Divide
DEFINE_CUDA_ELEMENTWISE_OP(Divide)

}  // namespace pten

using float16 = paddle::platform::float16;
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_CTX_KERNEL(add,
                       GPU,
                       ALL_LAYOUT,
                       pten::Add,
                       float,
                       double,
                       int,
                       int64_t,
                       float16,
                       complex64,
                       complex128) {}
PT_REGISTER_CTX_KERNEL(subtract,
                       GPU,
                       ALL_LAYOUT,
                       pten::Subtract,
                       float,
                       double,
                       int,
                       int64_t,
                       float16,
                       complex64,
                       complex128) {}
PT_REGISTER_CTX_KERNEL(divide,
                       GPU,
                       ALL_LAYOUT,
                       pten::Divide,
                       float,
                       double,
                       int,
                       int64_t,
                       float16,
                       complex64,
                       complex128) {}
PT_REGISTER_CTX_KERNEL(multiply,
                       GPU,
                       ALL_LAYOUT,
                       pten::Multiply,
                       float,
                       double,
                       int,
                       int64_t,
                       bool,
                       float16,
                       complex64,
                       complex128) {}
