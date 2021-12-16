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
#include "paddle/pten/kernels/hybird/eigen/reduce.h"
#include "paddle/pten/kernels/hybird/eigen/sign.h"
#include "paddle/pten/kernels/hybird/general/reduce_impl.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T>
void Sign(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  eigen::Sign<CPUContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CPUContext& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DataType in_dtype,
          DataType out_dtype,
          DenseTensor* out) {
  pten::general::Reduce<CPUContext, T, pten::eigen::MeanFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T>
void Sum(const CPUContext& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType in_dtype,
         DataType out_dtype,
         DenseTensor* out) {
  pten::general::Reduce<CPUContext, T, pten::eigen::SumFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

}  // namespace pten

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::paddle::platform::bfloat16;
PT_REGISTER_KERNEL(sign, CPU, ALL_LAYOUT, pten::Sign, float, double) {}
PT_REGISTER_KERNEL(mean, CPU, ALL_LAYOUT, pten::Mean, float, double, bool) {}
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
