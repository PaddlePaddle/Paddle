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

#include "paddle/phi/kernels/maxout_grad_kernel.h"

#include "paddle/fluid/operators/math/maxouting.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void MaxOutGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out,
                      const DenseTensor& out_grad,
                      int groups,
                      int axis,
                      DenseTensor* x_grad) {
  if (axis < 0) {
    axis += x.dims().size();
  }

  phi::funcs::SetConstant<Context, T> zero;
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    zero(dev_ctx, x_grad, static_cast<T>(0.0));
    paddle::operators::math::MaxOutGradFunctor<Context, T> maxout_backward;
    maxout_backward(dev_ctx, x, x_grad, out, out_grad, groups, axis);
  }
}

}  // namespace phi
