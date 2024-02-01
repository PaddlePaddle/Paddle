// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/binomial_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/binomial_functor.h"

namespace phi {

template <typename T, typename Context>
void BinomialKernel(const Context& ctx,
                    const DenseTensor& count,
                    const DenseTensor& prob,
                    DenseTensor* out) {
  auto numel = count.numel();
  auto* count_data = count.data<T>();
  auto* prob_data = prob.data<T>();
  int64_t* out_data = ctx.template Alloc<int64_t>(out);

  for (int64_t i = 0; i < numel; ++i) {
    out_data[i] = funcs::BinomialFunctor<T>(ctx, count_data[i], prob_data[i]);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    binomial, CPU, ALL_LAYOUT, phi::BinomialKernel, float, double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
