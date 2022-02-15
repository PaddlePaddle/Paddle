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

#pragma once

#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/abs_grad_kernel.h"

namespace pten {

template <typename T, typename Context>
void AbsGradKernel(const Context& ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   DenseTensor* dx) {
  auto numel = dout.numel();
  auto* dout_data = dout.data<paddle::operators::math::Real<T>>();
  auto* x_data = x.data<T>();

  ctx.template Alloc<T>(dx, static_cast<size_t>(numel * sizeof(T)));
  auto* dx_data = dx->data<T>();

  paddle::platform::ForRange<Context> for_range(ctx, numel);
  paddle::operators::math::AbsGradFunctor<T> functor(
      dout_data, x_data, dx_data, numel);
  for_range(functor);
}

template <typename T, typename Context>
void AbsDoubleGradKernel(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& ddx,
                         DenseTensor* ddout) {
  auto numel = ddx.numel();
  auto* ddx_data = ddx.data<T>();
  auto* x_data = x.data<T>();
  ctx.template Alloc<T>(ddout, static_cast<size_t>(numel * sizeof(T)));
  auto* ddout_data = ddout->data<T>();

  paddle::platform::ForRange<Context> for_range(ctx, numel);
  paddle::operators::math::AbsGradGradFunctor<T> functor(
      ddx_data, x_data, ddout_data, numel);
  for_range(functor);
}

}  // namespace pten
