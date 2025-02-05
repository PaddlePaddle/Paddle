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

#include <algorithm>
#include <vector>

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#define MAX_RANK_SUPPORTED 8

namespace phi {
using Tensor = DenseTensor;

template <typename Context, typename T, int Rank>
void Expand(const Context& ctx,
            const DenseTensor& x,
            const IntArray& shape,
            DenseTensor* out) {
  auto in_dims = x.dims();
  auto expand_shape = shape.GetData();
  auto vec_in_dims = common::vectorize<int64_t>(in_dims);
  auto diff = expand_shape.size() - vec_in_dims.size();
  vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
  std::vector<int> repeat_times(vec_in_dims.size());
  if (Rank == 0) {
    phi::Copy<Context>(ctx, x, ctx.GetPlace(), false, out);
    return;
  }
  for (size_t i = 0; i < vec_in_dims.size(); ++i) {
    if (i < diff) {
      PADDLE_ENFORCE_GE(
          expand_shape[i],
          0,
          common::errors::InvalidArgument(
              "The expanded size (%d) for non-existing dimensions must be "
              "positive for expand_v2 op.",
              expand_shape[i]));
      repeat_times[i] = expand_shape[i];
    } else if (expand_shape[i] == 0) {
      PADDLE_ENFORCE_EQ(
          vec_in_dims[i] == 1 || vec_in_dims[i] == expand_shape[i],
          true,
          common::errors::InvalidArgument(
              "The value (%d) of the non-singleton dimension does not match"
              " the corresponding value (%d) in shape for expand_v2 op.",
              vec_in_dims[i],
              expand_shape[i]));
      repeat_times[i] = 0;
    } else if (expand_shape[i] > 0) {
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i],
            expand_shape[i],
            common::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in shape for expand_v2 op.",
                vec_in_dims[i],
                expand_shape[i]));
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = expand_shape[i];
      }
    } else if (expand_shape[i] == -1) {
      repeat_times[i] = 1;
    }
  }
  Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    bcast_dims[i] = repeat_times[i];
  }

  DDim new_in_dims = common::make_ddim(vec_in_dims);
  DDim out_dims(new_in_dims);
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    if (repeat_times[i] == 0) {
      out_dims[i] = 0;
    } else if (expand_shape[i] == -1) {
      out_dims[i] = new_in_dims[i];
    } else {
      out_dims[i] *= repeat_times[i];
    }
  }

  out->Resize(out_dims);
  auto x0 = EigenTensor<T, Rank>::From(x, new_in_dims);
  ctx.template Alloc<T>(out);
  out->data<T>();

  auto y = EigenTensor<T, Rank>::From(*out, out_dims);
  auto& place = *ctx.eigen_device();
  // use 32-bit index to speed up
  bool use_32bit_index = y.size() < Eigen::NumTraits<int>::highest();
  if (use_32bit_index) {
    phi::funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, To32BitIndex(y), To32BitIndex(x0), bcast_dims);
  } else {
    phi::funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
        place, y, x0, bcast_dims);
  }
}

template <typename T, typename Context>
void ExpandKernel(const Context& ctx,
                  const DenseTensor& x,
                  const IntArray& shape,
                  DenseTensor* out) {
  auto rank = x.dims().size();
  auto expand_shape = shape.GetData();
  auto shape_size = expand_shape.size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      common::errors::InvalidArgument(
          "The rank of the input 'X' for expand_v2 op must be positive, "
          "but the value received is %d.",
          rank));
  PADDLE_ENFORCE_LE(
      rank,
      MAX_RANK_SUPPORTED,
      common::errors::InvalidArgument(
          "The rank of the input 'X' for expand_v2 op must be less than "
          "or equal to %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          rank));
  PADDLE_ENFORCE_GE(
      shape_size,
      rank,
      common::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2 op must be "
          "greater than or equal to the rank (%d) of the input 'X'.",
          shape_size,
          rank));
  PADDLE_ENFORCE_LE(
      shape_size,
      MAX_RANK_SUPPORTED,
      common::errors::InvalidArgument(
          "The number (%d) of elements of 'shape' for expand_v2 op must be "
          "less than or equal to %d.",
          shape_size,
          MAX_RANK_SUPPORTED));
  rank = std::max(rank, static_cast<int>(shape_size));
  switch (rank) {
    case 0:
      Expand<Context, T, 0>(ctx, x, shape, out);
      break;
    case 1:
      Expand<Context, T, 1>(ctx, x, shape, out);
      break;
    case 2:
      Expand<Context, T, 2>(ctx, x, shape, out);
      break;
    case 3:
      Expand<Context, T, 3>(ctx, x, shape, out);
      break;
    case 4:
      Expand<Context, T, 4>(ctx, x, shape, out);
      break;
    case 5:
      Expand<Context, T, 5>(ctx, x, shape, out);
      break;
    case 6:
      Expand<Context, T, 6>(ctx, x, shape, out);
      break;
    case 7:
      Expand<Context, T, 7>(ctx, x, shape, out);
      break;
    case 8:
      Expand<Context, T, 8>(ctx, x, shape, out);
      break;
  }
}

}  // namespace phi
