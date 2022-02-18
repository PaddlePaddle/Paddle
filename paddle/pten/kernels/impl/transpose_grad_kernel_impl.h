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

#include "paddle/pten/kernels/funcs/math_function.h"
#include "paddle/pten/kernels/transpose_grad_kernel.h"
#include "paddle/pten/kernels/transpose_kernel.h"

namespace pten {

template <typename T, typename Context>
void TransposeKernelImpl(const Context& ctx,
                         const DenseTensor& x,
                         const std::vector<int>& axis,
                         DenseTensor* out) {
  ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  int rank = axis.size();
  switch (rank) {
    case 1:
      funcs::Transpose<Context, T, 1> trans1;
      trans1(ctx, x, out, axis);
      break;
    case 2:
      funcs::Transpose<Context, T, 2> trans2;
      trans2(ctx, x, out, axis);
      break;
    case 3:
      funcs::Transpose<Context, T, 3> trans3;
      trans3(ctx, x, out, axis);
      break;
    case 4:
      funcs::Transpose<Context, T, 4> trans4;
      trans4(ctx, x, out, axis);
      break;
    case 5:
      funcs::Transpose<Context, T, 5> trans5;
      trans5(ctx, x, out, axis);
      break;
    case 6:
      funcs::Transpose<Context, T, 6> trans6;
      trans6(ctx, x, out, axis);
      break;
    default:
      // for rank >= 7 situation
      funcs::TransposeNormal<Context, T> trans_normal;
      trans_normal(ctx, x, out, axis);
  }
}

template <typename T, typename Context>
void TransposeGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const std::vector<int>& axis,
                         DenseTensor* x_grad) {
  std::vector<int> reversed_axis(axis);

  dev_ctx.template Alloc<T>(x_grad);
  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }

  TransposeKernelImpl<T, Context>(dev_ctx, out_grad, reversed_axis, x_grad);
}

}  // namespace pten
