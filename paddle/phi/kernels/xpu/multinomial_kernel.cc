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
#include "paddle/phi/kernels/funcs/multinomial_kernel_helper.h"

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
  int64_t* out_data = dev_ctx.template Alloc<int64_t>(out);
  auto in_dims = x.dims();
  int64_t dim_size = in_dims.size();
  const int64_t num_categories = in_dims[dim_size - 1];
  const int64_t num_distributions = dim_size > 1 ? in_dims[dim_size - 2] : 1;
  int64_t seed = dev_ctx.GetGenerator()->Random64();

  // If replacement is False, it's not a replaceable sample. Every category
  // can be used only once.
  if (!replacement) {
    MultinomialInputChecker<T, Context>(dev_ctx, x, num_samples);
  }

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  const float* in_data = nullptr;
  if (!std::is_same<T, float>::value) {
    // multinomial only accept float as input
    using XPUType = typename XPUTypeTrait<T>::Type;
    auto numel = x.numel();
    float* cast_buffer = RAII_GUARD.alloc_l3_or_gm<float>(numel);
    int r =
        xpu::cast<XPUType, float>(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  cast_buffer,
                                  numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    in_data = cast_buffer;
  } else {
    in_data = reinterpret_cast<const float*>(x.data<T>());
  }

  // int multinomial(Context* ctx, const T* x, TID* y, int64_t num_samples,
  // int64_t num_categories, int64_t num_distributions, bool replacement,
  // int64_t seed);
  int r = xpu::multinomial<float, int64_t>(dev_ctx.x_context(),
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

PD_REGISTER_KERNEL(multinomial,
                   XPU,
                   ALL_LAYOUT,
                   phi::MultinomialKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
