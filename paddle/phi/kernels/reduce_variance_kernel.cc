// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/reduce_variance_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void VarianceKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int64_t>& dims,
                    bool keep_dim,
                    DenseTensor* out) {
  DenseTensor temp_mean = Mean<T, Context>(dev_ctx, x, dims, true);
  DenseTensor temp_differences = Subtract<T, Context>(dev_ctx, x, temp_mean);
  DenseTensor temp_pow =
      Multiply<T, Context>(dev_ctx, temp_differences, temp_differences);

  MeanKernel<T, Context>(dev_ctx, temp_pow, dims, keep_dim, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    variance, CPU, ALL_LAYOUT, phi::VarianceKernel, float, double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    variance, GPU, ALL_LAYOUT, phi::VarianceKernel, float, double) {}
#endif
