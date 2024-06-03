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

#include "paddle/phi/kernels/cumprod_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cumprod.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/exclusive_scan.h"
#include "paddle/phi/kernels/funcs/inclusive_scan.h"

namespace phi {

template <typename T, typename Context>
void CumprodKernel(const Context &dev_ctx,
                   const DenseTensor &input,
                   int dim,
                   bool exclusive,
                   bool reverse,
                   DenseTensor *out) {
  const auto *x = &input;
  auto *y = out;
  size_t outer_dim, mid_dim, inner_dim;
  GetCumprodDimInfo(x->dims(), dim, &outer_dim, &mid_dim, &inner_dim);
  if (x->dims().size() == 0) {
    phi::Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, out);
    return;
  }

  const auto *x_data = x->data<T>();
  auto *y_data = dev_ctx.template Alloc<T>(y);
  if (!exclusive) {
    phi::funcs::InclusiveScan(x_data,
                              y_data,
                              outer_dim,
                              mid_dim,
                              inner_dim,
                              static_cast<T>(1),
                              funcs::MultiplyFunctor<T>(),
                              /*reverse=*/reverse,
                              dev_ctx);
  } else {
    phi::funcs::ExclusiveScan(x_data,
                              y_data,
                              outer_dim,
                              mid_dim,
                              inner_dim,
                              static_cast<T>(1),
                              funcs::MultiplyFunctor<T>(),
                              /*reverse=*/reverse,
                              dev_ctx);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cumprod,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumprodKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
