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

#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  ctx.template Alloc<phi::dtype::Real<T>>(
      out, size_t(x.numel() * sizeof(phi::dtype::Real<T>)));
  auto* out_data = out->data<phi::dtype::Real<T>>();

  phi::funcs::ForRange<Context> for_range(ctx, numel);
  phi::funcs::AbsFunctor<T> functor(x_data, out_data, numel);
  for_range(functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(abs,
                   CPU,
                   ALL_LAYOUT,
                   phi::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
