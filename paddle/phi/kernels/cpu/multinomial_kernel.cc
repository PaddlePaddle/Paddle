/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/multinomial_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/multinomial_functor.h"

namespace phi {

template <typename T, typename Context>
void MultinomialKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const Scalar& num_samples,
                       bool replacement,
                       DenseTensor* out) {
  auto* in_data = x.data<T>();
  int64_t* out_data = dev_ctx.template Alloc<int64_t>(out);
  auto in_dims = x.dims();
  int64_t in_rank = in_dims.size();
  const int64_t num_categories = in_dims[in_rank - 1];
  const int64_t num_distributions = in_rank > 1 ? in_dims[in_rank - 2] : 1;

  funcs::MultinomialFunctor<T>(dev_ctx,
                               out_data,
                               in_data,
                               num_samples.to<int>(),
                               replacement,
                               num_categories,
                               num_distributions);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    multinomial, CPU, ALL_LAYOUT, phi::MultinomialKernel, float, double) {}
