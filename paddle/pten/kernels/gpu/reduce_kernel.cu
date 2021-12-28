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

#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/gpu/reduce.h"

namespace pten {

template <typename T, typename ContextT>
void MeanKernel(const ContextT& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                bool reduce_all,
                DenseTensor* out) {
  auto out_dtype = x.dtype();
  pten::Reduce<T, kps::AddFunctor, kps::DivideFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename ContextT>
void SumKernel(const ContextT& dev_ctx,
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

PT_REGISTER_CTX_KERNEL(
    mean, GPU, ALL_LAYOUT, pten::MeanKernel, float, double, bool, float16) {}

PT_REGISTER_CTX_KERNEL(sum,
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
