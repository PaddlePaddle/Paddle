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

#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
inline void TransCompute(const int dim,
                         const Context& dev_ctx,
                         const DenseTensor& in,
                         DenseTensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      Transpose<Context, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      Transpose<Context, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      Transpose<Context, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      Transpose<Context, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      Transpose<Context, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      Transpose<Context, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      TransposeNormal<Context, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

}  // namespace funcs
}  // namespace phi
