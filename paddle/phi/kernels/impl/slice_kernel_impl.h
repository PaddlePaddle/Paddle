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

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

template <typename T, typename Context, size_t D>
void SliceCompute(const Context& ctx,
                  const DenseTensor& input,
                  const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& starts_t,
                  const std::vector<int64_t>& ends_t,
                  const std::vector<int64_t>& infer_flags,
                  const std::vector<int64_t>& decrease_axis,
                  DenseTensor* out) {
  // Step 1: Get the accurate attribute value of starts and ends
  std::vector<int64_t> starts = starts_t;
  std::vector<int64_t> ends = ends_t;
  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      phi::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  // Step 2: Compute output
  auto in = &input;

  auto in_dims = in->dims();
  auto out_dims = out->dims();
  auto slice_dims = out_dims;

  // 2.1 Infer output dims
  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  funcs::CheckAndUpdateSliceAttrs<int64_t>(in_dims, axes, &starts, &ends);
  slice_dims = funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  out_dims = funcs::GetDecreasedDims<int64_t>(slice_dims, decrease_axis);

  // 2.2 Get output
  auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();

  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = slice_dims[i];
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    offsets[axes[i]] = starts[i];
  }

  out->Resize(slice_dims);
  ctx.template Alloc<T>(out);

  auto in_t = EigenTensor<T, D>::From(*in, in_dims);
  auto out_t = EigenTensor<T, D>::From(*out, slice_dims);
  auto& eigen_place = *ctx.eigen_device();

  if (in->numel() <= Eigen::NumTraits<int>::highest()) {
    // similar to tf.slice:
    // if element number less than INT_MAX, change the type of index to int
    Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
    for (size_t i = 0; i < D; i++) {
      offsets_32bit[i] = offsets[i];
      extents_32bit[i] = extents[i];
    }
    funcs::EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place,
        To32BitIndex(out_t),
        To32BitIndex(in_t),
        offsets_32bit,
        extents_32bit);
  } else {
    funcs::EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
        eigen_place, out_t, in_t, offsets, extents);
  }

  out->Resize(out_dims);
}

template <typename T, typename Context>
void SliceRawKernel(const Context& ctx,
                    const DenseTensor& input,
                    const std::vector<int64_t>& axes,
                    const ScalarArray& starts_arr,
                    const ScalarArray& ends_arr,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    DenseTensor* out) {
  int rank = input.dims().size();

  auto& starts = starts_arr.GetData();
  auto& ends = ends_arr.GetData();

  switch (rank) {
    case 1:
      SliceCompute<T, Context, 1>(
          ctx, input, axes, starts, ends, infer_flags, decrease_axis, out);
      break;
    case 2:
      SliceCompute<T, Context, 2>(
          ctx, input, axes, starts, ends, infer_flags, decrease_axis, out);
      break;
    case 3:
      SliceCompute<T, Context, 3>(
          ctx, input, axes, starts, ends, infer_flags, decrease_axis, out);
      break;
    case 4:
      SliceCompute<T, Context, 4>(
          ctx, input, axes, starts, ends, infer_flags, decrease_axis, out);
      break;
    case 5:
      SliceCompute<T, Context, 5>(
          ctx, input, axes, starts, ends, infer_flags, decrease_axis, out);
      break;
    case 6:
      SliceCompute<T, Context, 6>(
          ctx, input, axes, starts, ends, infer_flags, decrease_axis, out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

}  // namespace phi
