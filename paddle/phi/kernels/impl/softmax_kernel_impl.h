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

#include "paddle/phi/kernels/softmax_kernel.h"

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      int axis,
                      DenseTensor* out) {
  const int rank = x.dims().size();
  const int calc_axis = phi::funcs::CanonicalAxis(axis, rank);
  int axis_dim = x.dims()[calc_axis];

  // allocate memory on device.
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  const int n = phi::funcs::SizeToAxis(calc_axis, x.dims());
  const int d = phi::funcs::SizeFromAxis(calc_axis, x.dims());
  DenseTensor X_2d, Out_2d;
  X_2d.ShareDataWith(x).Resize({n, d});
  Out_2d.ShareDataWith(*out).Resize({n, d});
  paddle::operators::math::SoftmaxFunctor<Context, T, false>()(
      dev_ctx, axis_dim, &X_2d, &Out_2d);
}

}  // namespace phi
