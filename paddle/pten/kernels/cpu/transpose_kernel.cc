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

#include "paddle/pten/kernels/transpose_kernel.h"
#include <vector>
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"

namespace pten {

template <typename T, typename Context>
void TransposeKernel(const Context& ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  out->mutable_data<T>(ctx.GetPlace());
  int rank = axis.size();
  switch (rank) {
    case 1:
      paddle::operators::math::Transpose<Context, T, 1> trans1;
      trans1(ctx, x, out, axis);
      break;
    case 2:
      paddle::operators::math::Transpose<Context, T, 2> trans2;
      trans2(ctx, x, out, axis);
      break;
    case 3:
      paddle::operators::math::Transpose<Context, T, 3> trans3;
      trans3(ctx, x, out, axis);
      break;
    case 4:
      paddle::operators::math::Transpose<Context, T, 4> trans4;
      trans4(ctx, x, out, axis);
      break;
    case 5:
      paddle::operators::math::Transpose<Context, T, 5> trans5;
      trans5(ctx, x, out, axis);
      break;
    case 6:
      paddle::operators::math::Transpose<Context, T, 6> trans6;
      trans6(ctx, x, out, axis);
      break;
    default:
      // for rank >= 7 situation
      paddle::operators::math::TransposeNormal<Context, T> trans_normal;
      trans_normal(ctx, x, out, axis);
  }
}
}  // namespace pten

PT_REGISTER_KERNEL(transpose,
                   CPU,
                   ALL_LAYOUT,
                   pten::TransposeKernel,
                   bool,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>) {}
