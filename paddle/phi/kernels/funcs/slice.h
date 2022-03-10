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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

// TODO(paddle-dev): Remove this file when we can call related Kernel directly

namespace phi {
namespace funcs {

template <typename Context, typename T, size_t D>
void EigenSliceWrapper(const Context& dev_ctx,
                       const DenseTensor* in,
                       const std::vector<int>& start,
                       const std::vector<int>& end,
                       DenseTensor* out) {
  // Slice by call Eigen Tensor Function `.slice()`
  size_t rank = in->dims().size();
  PADDLE_ENFORCE_EQ(start.size(),
                    rank,
                    errors::InvalidArgument(
                        "EigenSliceWrapper function start "
                        "argument must have the same length as input rank."));
  PADDLE_ENFORCE_EQ(end.size(),
                    rank,
                    errors::InvalidArgument(
                        "EigenSliceWrapper function end "
                        "argument must have the same length as input rank."));
  auto eigen_place_ptr = dev_ctx.eigen_device();
  auto eigen_place = *eigen_place_ptr;
  auto out_t = phi::EigenTensor<T, D>::From(*out, out->dims());
  auto in_t = phi::EigenTensor<T, D>::From(*in, in->dims());
  Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
  for (size_t i = 0; i < D; i++) {
    offsets_32bit[i] = start[i];
    extents_32bit[i] = end[i];
  }
  EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
      eigen_place,
      phi::To32BitIndex(out_t),
      phi::To32BitIndex(in_t),
      offsets_32bit,
      extents_32bit);
}

#define SLICE_RANK_CASE(N)                                                \
  case N: {                                                               \
    EigenSliceWrapper<Context, T, N>(dev_ctx, &x, offset, extends, &ret); \
    break;                                                                \
  }

template <typename T, typename Context>
DenseTensor Slice(const Context& dev_ctx,
                  const DenseTensor& x,
                  std::vector<int> axes,
                  std::vector<int> starts,
                  std::vector<int> ends) {
  DenseTensor ret;
  std::vector<int> new_axes = axes;
  std::vector<int> out_shape = phi::vectorize<int>(x.dims());
  size_t rank = out_shape.size();
  PADDLE_ENFORCE_EQ(
      axes.size(),
      starts.size(),
      errors::InvalidArgument("Slice Operator Argument Invalided"));
  PADDLE_ENFORCE_EQ(
      ends.size(),
      starts.size(),
      errors::InvalidArgument("Slice Operator Argument Invalided"));
  for (unsigned int i = 0; i < axes.size(); ++i) {
    int axis = axes[i];
    if (axis < 0) axis = rank + axis;
    new_axes[i] = axis;  // change negative to positive
    int st = starts[i];
    int ed = ends[i];
    PADDLE_ENFORCE_GT(
        ed,
        st,
        errors::InvalidArgument("C++ Slice Operation Not Support End < Start"));
    out_shape[axis] = ed - st;
  }
  std::vector<int> offset(rank), extends(rank);
  for (size_t i = 0; i < rank; ++i) {
    offset[i] = 0;
    extends[i] = x.dims()[i];
  }
  for (size_t i = 0; i < new_axes.size(); ++i) {
    offset[new_axes[i]] = starts[i];
    extends[new_axes[i]] = ends[i] - starts[i];
  }
  ret.Resize(phi::make_ddim(out_shape));
  dev_ctx.template Alloc<T>(&ret);
  switch (rank) {
    SLICE_RANK_CASE(1);
    SLICE_RANK_CASE(2);
    SLICE_RANK_CASE(3);
    SLICE_RANK_CASE(4);
    SLICE_RANK_CASE(5);
    SLICE_RANK_CASE(6);
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
