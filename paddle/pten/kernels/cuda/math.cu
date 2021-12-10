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

#include "paddle/pten/kernels/cuda/math.h"

#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/pten/kernels/hybird/cuda/elementwise/elementwise.h"
#include "paddle/pten/kernels/hybird/cuda/reduce/reduce.h"
#include "paddle/pten/kernels/hybird/eigen/scale.h"
#include "paddle/pten/kernels/hybird/eigen/sign.h"
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
void Sign(const CUDAContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  eigen::Sign<CUDAContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CUDAContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DataType in_dtype,
          DataType out_dtype,
          DenseTensor* out) {
  pten::Reduce<T, paddle::operators::CustomMean>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T>
void Scale(const CUDAContext& dev_ctx,
           const DenseTensor& x,
           const Scalar& scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  eigen::Scale<CUDAContext, T>(
      dev_ctx, x, scale.to<float>(), bias, bias_after_scale, out);
}

// Create the definition of ElementwiseAdd
DEFINE_CUDA_ELEMENTWISE_OP(Add)
// Create the definition of ElementwiseSub
DEFINE_CUDA_ELEMENTWISE_OP(Sub)
// Create the definition of ElementwiseMul
DEFINE_CUDA_ELEMENTWISE_OP(Mul)
// Create the definition of ElementwiseDiv
DEFINE_CUDA_ELEMENTWISE_OP(Div)

template <typename T>
void Sum(const CUDAContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType in_dtype,
         DataType out_dtype,
         DenseTensor* out) {
  pten::Reduce<T, paddle::operators::CustomSum>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

}  // namespace pten

using float16 = paddle::platform::float16;
using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_KERNEL(sign, CUDA, ANY, pten::Sign, float, double, float16) {}
PT_REGISTER_KERNEL(mean, CUDA, ANY, pten::Mean, float, double, bool) {}
PT_REGISTER_KERNEL(scale,
                   CUDA,
                   ANY,
                   pten::Scale,
                   float,
                   double,
                   float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL(add,
                   CUDA,
                   ANY,
                   pten::ElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(subtract,
                   CUDA,
                   ANY,
                   pten::ElementwiseSub,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(divide,
                   CUDA,
                   ANY,
                   pten::ElementwiseDiv,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(multiply,
                   CUDA,
                   ANY,
                   pten::ElementwiseMul,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   float16,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(sum,
                   CUDA,
                   ANY,
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
