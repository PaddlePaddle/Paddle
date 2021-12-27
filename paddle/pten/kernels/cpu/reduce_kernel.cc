// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/kernels/reduce_kernel.h"

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/cpu/reduce.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T, typename ContextT>
void Mean(const ContextT& dev_ctx,
          const DenseTensor& x,
          const std::vector<int64_t>& dims,
          bool keep_dim,
          bool reduce_all,
          DenseTensor* out) {
  auto out_dtype = x.dtype();
  pten::Reduce<ContextT, T, pten::eigen::MeanFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename ContextT>
void Sum(const ContextT& dev_ctx,
         const DenseTensor& x,
         const std::vector<int64_t>& dims,
         bool keep_dim,
         bool reduce_all,
         DataType out_dtype,
         DenseTensor* out) {
  pten::Reduce<ContextT, T, pten::eigen::SumFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

}  // namespace pten

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

PT_REGISTER_CTX_KERNEL(mean, CPU, ALL_LAYOUT, pten::Mean, float, double, bool) {
}
PT_REGISTER_CTX_KERNEL(sum,
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
