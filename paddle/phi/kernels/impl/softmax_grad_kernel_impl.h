/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/softmax.h"
#include "paddle/phi/kernels/softmax_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const DenseTensor& out,
                       const DenseTensor& out_grad,
                       int axis,
                       DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const int rank = x_grad->dims().size();
  // For 0D Tensor
  if (rank == 0) {
    phi::funcs::set_constant(dev_ctx, x_grad, static_cast<T>(0.0));
    return;
  }
  // For zero-sized Tensor
  if (x_grad->numel() == 0) {
    return;
  }

  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x_grad->dims()[calc_axis];

  const int n = phi::funcs::SizeToAxis(calc_axis, x_grad->dims());
  const int d = phi::funcs::SizeFromAxis(calc_axis, x_grad->dims());
  DenseTensor dX_2d, Out_2d, dOut_2d;
  dX_2d.ShareDataWith(*x_grad).Resize({n, d});
  Out_2d.ShareDataWith(out).Resize({n, d});
  dOut_2d.ShareDataWith(out_grad).Resize({n, d});

  phi::funcs::SoftmaxGradFunctor<Context, T>()(
      dev_ctx, axis_dim, &Out_2d, &dOut_2d, &dX_2d);
}

}  // namespace phi
