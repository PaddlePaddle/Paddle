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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename Context>
void AngleGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad) {
  auto numel = out_grad.numel();
  auto* dout_data = out_grad.data<phi::dtype::Real<T>>();
  auto* x_data = x.data<T>();
  x_grad->Resize(out_grad.dims());
  auto* dx_data = dev_ctx.template Alloc<T>(x_grad);

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::AngleGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
  for_range(functor);
}

}  // namespace phi
