// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/multinomial_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MultinomialKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const Scalar& num_samples,
                       bool replacement,
                       DenseTensor* out) {
  auto int_num_samples = num_samples.to<int64_t>();
  auto* in_data = x.data<T>();
  int64_t* out_data = dev_ctx.template Alloc<int64_t>(out);
  auto in_dims = x.dims();
  int64_t dim_size = in_dims.size();
  const int64_t num_categories = in_dims[dim_size - 1];
  const int64_t num_distributions = dim_size > 1 ? in_dims[dim_size - 2] : 1;
  int64_t seed = dev_ctx.GetGenerator()->Random64();

  // int multinomial(Context* ctx, const T* x, TID* y, int64_t num_samples,
  // int64_t num_categories, int64_t num_distributions, bool replacement,
  // int64_t seed);
  int r = xpu::multinomial<T, int64_t>(dev_ctx.x_context(),
                                       in_data,
                                       out_data,
                                       int_num_samples,
                                       num_categories,
                                       num_distributions,
                                       replacement,
                                       seed);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "multinomial");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    multinomial, XPU, ALL_LAYOUT, phi::MultinomialKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
