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

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

// TODO(paddle-dev): Remove this file when we can call related Kernel directly

namespace phi {
namespace funcs {

#define RANSPOSE_RANK_CASE(N)                         \
  case N: {                                           \
    phi::funcs::Transpose<DeviceContext, T, N> trans; \
    trans(dev_ctx, x, &ret, axis);                    \
    break;                                            \
  }

template <typename T, typename Context>
DenseTensor TransposeLast2Dims(const Context& dev_ctx, const DenseTensor& x) {
  // transpose the last two dimision
  DenseTensor ret;
  auto x_dim = x.dims();
  auto x_vec = phi::vectorize<int>(x_dim);
  int rank = x_vec.size();
  std::swap(x_vec[rank - 1], x_vec[rank - 2]);
  std::vector<int> out_shape = x_vec;
  std::vector<int> axis(rank);
  for (int i = 0; i < rank; ++i) {
    axis[i] = i;
  }
  std::swap(axis[rank - 1], axis[rank - 2]);
  ret.Resize(phi::make_ddim(x_vec));
  ctx.template Alloc<T>(&ret);
  switch (rank) {
    RANSPOSE_RANK_CASE(2);
    RANSPOSE_RANK_CASE(3);
    RANSPOSE_RANK_CASE(4);
    RANSPOSE_RANK_CASE(5);
    RANSPOSE_RANK_CASE(6);
    default: {
      PADDLE_THROW(
          errors::InvalidArgument("Invalid Rank number, "
                                  "currently only support rank between 2~6"));
    }
  }
  return ret;
}

}  // namespace funcs
}  // namespace phi
