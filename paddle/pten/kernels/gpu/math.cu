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

#include "paddle/pten/kernels/gpu/math.h"

#include "paddle/pten/kernels/hybird/cuda/elementwise/elementwise.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce.h"
#include "paddle/pten/kernels/hybird/general/elementwise_functor.h"
#include "paddle/pten/kernels/hybird/general/reduce_impl.h"

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

namespace kps = paddle::operators::kernel_primitives;

namespace pten {

/**
 * Util Functors
 */

template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n)
      : n_inv(static_cast<T>(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};

/**
 * Kernels
 */

template <typename T>
void Mean(const GPUContext& dev_ctx,
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

template <typename T>
void Sum(const GPUContext& dev_ctx,
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
    mean, GPU, ALL_LAYOUT, pten::Mean, float, double, bool, float16) {}
PT_REGISTER_KERNEL(add,
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
PT_REGISTER_KERNEL(subtract,
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
PT_REGISTER_KERNEL(divide,
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
PT_REGISTER_KERNEL(multiply,
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
PT_REGISTER_KERNEL(sum,
                   GPU,
                   ALL_LAYOUT,
                   pten::Sum,
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
