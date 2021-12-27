//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/cpu/math.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/kernels/hybird/cpu/elementwise.h"
#include "paddle/pten/kernels/hybird/eigen/reduce.h"
#include "paddle/pten/kernels/hybird/general/elementwise_functor.h"
#include "paddle/pten/kernels/hybird/general/reduce_impl.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T>
void Mean(const CPUContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DenseTensor* out) {
  auto out_dtype = x.dtype();
  pten::general::Reduce<CPUContext, T, pten::eigen::MeanFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T>
void Divide(const CPUContext& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            int axis,
            DenseTensor* out) {
  // allocate memory for out
  out->mutable_data<T>();
  if (x.dims() == y.dims() && std::is_floating_point<T>::value) {
    SameDimsElementwiseCompute<general::SameDimsDivideFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseCompute<general::DivideFunctor<T>, T>(
          dev_ctx, x, y, axis, general::DivideFunctor<T>(), out);
    } else {
      ElementwiseCompute<general::InverseDivideFunctor<T>, T>(
          dev_ctx, x, y, axis, general::InverseDivideFunctor<T>(), out);
    }
  }
}

template <typename T>
void Sum(const CPUContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType out_dtype,
         DenseTensor* out) {
  pten::general::Reduce<CPUContext, T, pten::eigen::SumFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

// Create the definition of Add
DEFINE_CPU_ELEMENTWISE_OP(Add)

// Create the definition of Subtract
DEFINE_CPU_ELEMENTWISE_OP(Subtract)

// Create the definition of Multiply
DEFINE_CPU_ELEMENTWISE_OP(Multiply)

}  // namespace pten

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::paddle::platform::bfloat16;
PT_REGISTER_KERNEL(mean, CPU, ALL_LAYOUT, pten::Mean, float, double, bool) {}
PT_REGISTER_KERNEL(add,
                   CPU,
                   ALL_LAYOUT,
                   pten::Add,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(subtract,
                   CPU,
                   ALL_LAYOUT,
                   pten::Subtract,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(divide,
                   CPU,
                   ALL_LAYOUT,
                   pten::Divide,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(multiply,
                   CPU,
                   ALL_LAYOUT,
                   pten::Multiply,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL(sum,
                   CPU,
                   ALL_LAYOUT,
                   pten::Sum,
                   bool,
                   float,
                   double,
                   paddle::platform::float16,
                   int,
                   int64_t,
                   complex64,
                   complex128) {
  kernel->OutputAt(0).SetDataType(paddle::experimental::DataType::UNDEFINED);
}
