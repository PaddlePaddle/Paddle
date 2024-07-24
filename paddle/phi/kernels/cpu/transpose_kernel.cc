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

#include "paddle/phi/kernels/transpose_kernel.h"

#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  size_t x_rank = x.dims().size();
  std::vector<int> formatted_axis = axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = static_cast<int>(axis[i] + x_rank);
    }
  }

  ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  int rank = static_cast<int>(formatted_axis.size());
  switch (rank) {
    case 0:
      phi::Copy<Context>(ctx, x, ctx.GetPlace(), false, out);
      break;
    case 1:
      funcs::Transpose<Context, T, 1> trans1;
      trans1(ctx, x, out, formatted_axis);
      break;
    case 2:
      funcs::Transpose<Context, T, 2> trans2;
      trans2(ctx, x, out, formatted_axis);
      break;
    case 3:
      funcs::Transpose<Context, T, 3> trans3;
      trans3(ctx, x, out, formatted_axis);
      break;
    case 4:
      funcs::Transpose<Context, T, 4> trans4;
      trans4(ctx, x, out, formatted_axis);
      break;
    case 5:
      funcs::Transpose<Context, T, 5> trans5;
      trans5(ctx, x, out, formatted_axis);
      break;
    case 6:
      funcs::Transpose<Context, T, 6> trans6;
      trans6(ctx, x, out, formatted_axis);
      break;
    default:
      // for rank >= 7 situation
      funcs::TransposeNormal<Context, T> trans_normal;
      trans_normal(ctx, x, out, formatted_axis);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   CPU,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   bool,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   phi::dtype::float8_e4m3fn,
                   phi::dtype::float8_e5m2) {}
