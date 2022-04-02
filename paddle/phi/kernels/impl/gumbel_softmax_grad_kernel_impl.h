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

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/operators/math/softmax_impl.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace phi {

template <typename T, typename Context>
void GumbelSoftmaxGradKernel(const Context& ctx,
                             const DenseTensor& out,
                             const DenseTensor& dout,
                             int axis,
                             DenseTensor* dx) {
  const int rank = dx->dims().size();
  axis = funcs::CanonicalAxis(axis, rank);
  int axis_dim = dx->dims()[axis];
  // allocate memory on device.

  ctx.template Alloc<T>(dx);
  if (dx->numel() == 0) {
    return;
  }

  const int size_to_axis = funcs::SizeToAxis(axis, dx->dims());
  const int size_from_axis = funcs::SizeFromAxis(axis, dx->dims());
  DenseTensor dx_2d(*dx), out_2d(out), dout_2d(dout);
  dx_2d.Resize({size_to_axis, size_from_axis});
  out_2d.Resize({size_to_axis, size_from_axis});
  dout_2d.Resize({size_to_axis, size_from_axis});
  paddle::operators::math::SoftmaxGradFunctor<Context, T>()(
      ctx, axis_dim, &out_2d, &dout_2d, &dx_2d);
}

}  // namespace phi
