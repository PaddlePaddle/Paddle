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

<<<<<<< HEAD
=======
#include "paddle/fluid/operators/math/inclusive_scan.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cumprod.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
<<<<<<< HEAD
#include "paddle/phi/kernels/funcs/inclusive_scan.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace phi {

template <typename T, typename Context>
void CumprodKernel(const Context &dev_ctx,
                   const DenseTensor &input,
                   int dim,
                   DenseTensor *out) {
  const auto *x = &input;
  auto *y = out;
  size_t outer_dim, mid_dim, inner_dim;
  GetCumprodDimInfo(x->dims(), dim, &outer_dim, &mid_dim, &inner_dim);
<<<<<<< HEAD
  if (x->dims().size() == 0) {
    phi::Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, out);
    return;
  }

  const auto *x_data = x->data<T>();
  auto *y_data = dev_ctx.template Alloc<T>(y);
  phi::funcs::InclusiveScan(x_data,
                            y_data,
                            outer_dim,
                            mid_dim,
                            inner_dim,
                            static_cast<T>(1),
                            funcs::MultiplyFunctor<T>(),
                            /*reverse=*/false,
                            dev_ctx);
=======

  const auto *x_data = x->data<T>();
  auto *y_data = dev_ctx.template Alloc<T>(y);
  paddle::operators::math::InclusiveScan(x_data,
                                         y_data,
                                         outer_dim,
                                         mid_dim,
                                         inner_dim,
                                         static_cast<T>(1),
                                         funcs::MultiplyFunctor<T>(),
                                         /*reverse=*/false,
                                         dev_ctx);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
