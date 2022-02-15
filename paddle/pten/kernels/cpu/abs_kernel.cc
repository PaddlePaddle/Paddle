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

#include "paddle/pten/kernels/abs_kernel.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/common/complex.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename T, typename Context>
void AbsKernel(const Context& ctx, const DenseTensor& x, DenseTensor* out) {
  auto numel = x.numel();
  auto* x_data = x.data<T>();
  ctx.template Alloc<paddle::operators::math::Real<T>>(
      out, size_t(x.numel() * sizeof(paddle::operators::math::Real<T>)));
  auto* out_data = out->data<paddle::operators::math::Real<T>>();

  paddle::platform::ForRange<Context> for_range(ctx, numel);
  paddle::operators::math::AbsFunctor<T> functor(x_data, out_data, numel);
  for_range(functor);
}

}  // namespace pten

PT_REGISTER_KERNEL(abs,
                   CPU,
                   ALL_LAYOUT,
                   pten::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   pten::dtype::complex<float>,
                   pten::dtype::complex<double>) {}
