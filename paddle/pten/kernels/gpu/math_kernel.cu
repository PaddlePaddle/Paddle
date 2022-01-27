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

#include "paddle/pten/kernels/math_kernel.h"

#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/kernels/funcs/elementwise_functor.h"
#include "paddle/pten/kernels/gpu/elementwise.h"
#include "paddle/pten/kernels/gpu/reduce.h"

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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                             \
  template <typename T, typename Context>                            \
  void name##Kernel(const Context& dev_ctx,                          \
                    const DenseTensor& x,                            \
                    const DenseTensor& y,                            \
                    int axis,                                        \
                    DenseTensor* out) {                              \
    std::vector<const DenseTensor*> inputs;                          \
    std::vector<DenseTensor*> outputs;                               \
    inputs.emplace_back(&x);                                         \
    inputs.emplace_back(&y);                                         \
    outputs.emplace_back(out);                                       \
    out->mutable_data<T>();                                          \
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(     \
        dev_ctx, inputs, &outputs, axis, funcs::name##Functor<T>()); \
  }

/**
 * Util Functors
 */

template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n)
      : n_inv(static_cast<T>(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T x) const { return x * n_inv; }

 private:
  T n_inv;
};

/**
 * Kernels
 */

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                bool reduce_all,
                DenseTensor* out) {
  auto out_dtype = x.dtype();
  pten::Reduce<T, kps::AddFunctor, kps::DivideFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

// Create the definition of Add
DEFINE_CUDA_ELEMENTWISE_OP(Add)
// Create the definition of Subtract
DEFINE_CUDA_ELEMENTWISE_OP(Subtract)
// Create the definition of Multiply
DEFINE_CUDA_ELEMENTWISE_OP(Multiply)
// Create the definition of Divide
DEFINE_CUDA_ELEMENTWISE_OP(Divide)

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               bool reduce_all,
               DataType out_dtype,
               DenseTensor* out) {
  pten::Reduce<T, kps::AddFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

}  // namespace pten

using float16 = paddle::platform::float16;
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL(
    mean, GPU, ALL_LAYOUT, pten::MeanKernel, float, double, bool, float16) {}
PT_REGISTER_KERNEL(add,
                   GPU,
                   ALL_LAYOUT,
                   pten::AddKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(subtract,
                   GPU,
                   ALL_LAYOUT,
                   pten::SubtractKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(divide,
                   GPU,
                   ALL_LAYOUT,
                   pten::DivideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(multiply,
                   GPU,
                   ALL_LAYOUT,
                   pten::MultiplyKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(sum,
                   GPU,
                   ALL_LAYOUT,
                   pten::SumKernel,
                   bool,
                   float,
                   double,
                   float16,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
