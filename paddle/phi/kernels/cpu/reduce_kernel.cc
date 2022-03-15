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

#include "paddle/phi/kernels/reduce_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"

namespace phi {

template <typename T, typename Context>
void ReduceProdKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::vector<int64_t>& dims,
                      bool keep_dim,
                      bool reduce_all,
                      DenseTensor* out) {
  auto out_dtype = x.dtype();
  phi::Reduce<CPUContext, T, phi::funcs::ProdFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  auto out_dtype = x.dtype();
  phi::Reduce<CPUContext, T, phi::funcs::MaxFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  auto out_dtype = x.dtype();
  phi::Reduce<CPUContext, T, phi::funcs::MinFunctor>(
      dev_ctx, x, reduce_all, dims, keep_dim, out_dtype, out);
}

template <typename T, typename Context>
void AllRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  phi::BoolReduceKernel<CPUContext, T, phi::funcs::AllFunctor>(
      dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void AnyRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out) {
  phi::BoolReduceKernel<CPUContext, T, phi::funcs::AnyFunctor>(
      dev_ctx, x, dims, keep_dim, reduce_all, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(reduce_prod,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReduceProdKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(
    max_raw, CPU, ALL_LAYOUT, phi::MaxRawKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    min_raw, CPU, ALL_LAYOUT, phi::MinRawKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(all_raw, CPU, ALL_LAYOUT, phi::AllRawKernel, bool) {}
PD_REGISTER_KERNEL(any_raw, CPU, ALL_LAYOUT, phi::AnyRawKernel, bool) {}
