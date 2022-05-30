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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {

// check whether the tensor with dimension of second can assign to the
// tensor with dimension of first
inline void CheckIsDimsMatch(const DDim& first, const DDim& second) {
  int ignore_axis1 = 0, ignore_axis2 = 0;
  for (; ignore_axis1 < first.size(); ++ignore_axis1) {
    if (first[ignore_axis1] != 1) {
      break;
    }
  }
  for (; ignore_axis2 < second.size(); ++ignore_axis2) {
    if (second[ignore_axis2] != 1) {
      break;
    }
  }

  if (second.size() == ignore_axis2) {
    // second tensor has only one value
    return;
  }

  if (first.size() - ignore_axis1 >= second.size() - ignore_axis2) {
    auto idx1 = first.size() - 1;
    auto idx2 = second.size() - 1;
    bool is_match = true;
    for (; idx2 >= ignore_axis2; idx2--) {
      if (first[idx1--] != second[idx2] && second[idx2] != 1) {
        is_match = false;
        break;
      }
    }
    if (is_match) {
      return;
    }
  }
  PADDLE_THROW(errors::InvalidArgument(
      "The shape of tensor assigned value must match the shape "
      "of target shape: %d, but now shape is %d.",
      second.to_str(),
      first.to_str()));
}

template <typename T, typename Context, size_t RANK>
void SetValueImpl(const Context& dev_ctx,
                  const DenseTensor& in,
                  const DenseTensor& value,
                  const IntArray& starts,
                  const IntArray& ends,
                  const IntArray& steps,
                  const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& decrease_axes,
                  const std::vector<int64_t>& none_axes,
                  DenseTensor* out) {
  auto in_dims = in.dims();
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  phi::funcs::CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);
  auto slice_dims = phi::funcs::GetSliceDims(
      in_dims, axes, starts_local, ends_local, &steps_local);
  auto decrease_slice_dims =
      phi::funcs::GetDecreasedDims(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;
  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;

    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    for (int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      if (decrease_axes_cur < decrease_axes.size() &&
          decrease_axes[decrease_axes_cur] == i) {
        decrease_axes_cur++;
      } else {
        slice_dims_with_none.push_back(slice_dims[i]);
      }
    }
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }

    slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
  }

  auto place = dev_ctx.GetPlace();
  auto& eigen_place = *dev_ctx.eigen_device();

  // Here copy data from input to avoid data loss at PE and Graph level.
  // TODO(liym27): Speed up in the future version.
  // - Q: Why don't call ShareDataWith to speed up?
  // - A: Because it's not supported to ShareDataWith on OP's input and output
  // https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-and-ShareBufferWith-are-prohibited-in-OP
  // - Q: Why don't delete Input, after all, the input and output are the same
  // Tensor at program level?
  // - A: If deleting Input, the graph will be complex, such as there will
  // be two ops points to the output in graph: op1 -> output <- set_value.
  // In this case, we have to find a way to handle the running order of
  // set_value is what we want.
  Copy(dev_ctx, in, place, false, out);

  DenseTensor slice_tensor =
      Empty<T>(dev_ctx, IntArray{slice_dims.Get(), slice_dims.size()});
  DenseTensor pad_tensor =
      Empty<T>(dev_ctx, IntArray{in_dims.Get(), in_dims.size()});

  auto pad_e = EigenTensor<T, RANK>::From(pad_tensor, in_dims);
  auto out_e = EigenTensor<T, RANK>::From(*out);
  auto slice_e = EigenTensor<T, RANK>::From(slice_tensor, slice_dims);

  // Step 1: Set the value of out at `_index` to zero
  slice_e.device(eigen_place) = slice_e.constant(T(0));

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, RANK>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, RANK>();
  auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, RANK>();

  for (size_t i = 0; i < RANK; ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = slice_dims[i];
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = starts_local[i];
    ends_indices[axis_index] = ends_local[i];
    strides_indices[axis_index] = steps_local[i];
    if (starts_local[i] ==
        ends_local[i]) {  // slice is empty, data will not be changed
      return;
    }
  }

  out_e.stridedSlice(starts_indices, ends_indices, strides_indices)
      .device(eigen_place) = slice_e;

  // Step 2: Set a tensor with the same shape as out tensor. And its data at
  // '_index' is the same as value, and data out of '_index' to zero

  // - Step 2.1 Set slice tensor with value

  // NOTE(liym27): [ Why resize slice_tensor here? ]
  // A: When do broadcasting on slice_tensor and value, the shape of
  // slice_tensor should be decreased dims.
  // e.g.
  //  x[:,0] = value
  // x's shape = [3, 4], value's shape = [3]
  // We get slice_dims = [3, 1],  decrease_slice_dims = [3]
  // If do broadcasting on Tensor with shape [3, 1] and [3], the result's
  // shape is [3, 3], which cross the border;
  // If do broadcasting on Tensor with shape [3] and [3], the result's shape
  // is [3], which is right.

  slice_tensor.Resize(slice_dims_for_assign);
  CheckIsDimsMatch(slice_dims_for_assign, value.dims());
  // ElementwiseComputeEx can do broadcasting
  funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
      dev_ctx,
      slice_tensor,
      value,
      -1,
      funcs::SubtractFunctor<T>(),
      &slice_tensor);

  slice_tensor.Resize(slice_dims);

  // - Step 2.2 Pad slice tensor with 0
  pad_e.device(eigen_place) = pad_e.constant(T(0));
  pad_e.stridedSlice(starts_indices, ends_indices, strides_indices)
      .device(eigen_place) = slice_e;

  // Step 3: Set out tensor with value
  out_e.device(eigen_place) = out_e - pad_e;
}

template <typename T, typename Context>
void SetTensorValueKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& value,
                          const IntArray& starts,
                          const IntArray& ends,
                          const IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          DenseTensor* out) {
  const int rank = x.dims().size();

  switch (rank) {
    case 1:
      SetValueImpl<T, Context, 1>(dev_ctx,
                                  x,
                                  value,
                                  starts,
                                  ends,
                                  steps,
                                  axes,
                                  decrease_axes,
                                  none_axes,
                                  out);
      break;
    case 2:
      SetValueImpl<T, Context, 2>(dev_ctx,
                                  x,
                                  value,
                                  starts,
                                  ends,
                                  steps,
                                  axes,
                                  decrease_axes,
                                  none_axes,
                                  out);
      break;
    case 3:
      SetValueImpl<T, Context, 3>(dev_ctx,
                                  x,
                                  value,
                                  starts,
                                  ends,
                                  steps,
                                  axes,
                                  decrease_axes,
                                  none_axes,
                                  out);
      break;
    case 4:
      SetValueImpl<T, Context, 4>(dev_ctx,
                                  x,
                                  value,
                                  starts,
                                  ends,
                                  steps,
                                  axes,
                                  decrease_axes,
                                  none_axes,
                                  out);
      break;
    case 5:
      SetValueImpl<T, Context, 5>(dev_ctx,
                                  x,
                                  value,
                                  starts,
                                  ends,
                                  steps,
                                  axes,
                                  decrease_axes,
                                  none_axes,
                                  out);
      break;
    case 6:
      SetValueImpl<T, Context, 6>(dev_ctx,
                                  x,
                                  value,
                                  starts,
                                  ends,
                                  steps,
                                  axes,
                                  decrease_axes,
                                  none_axes,
                                  out);
      break;
    default:
      PADDLE_THROW(errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

template <typename T, typename Context>
void SetValueKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const IntArray& starts,
                    const IntArray& ends,
                    const IntArray& steps,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& decrease_axes,
                    const std::vector<int64_t>& none_axes,
                    const std::vector<int64_t>& shape,
                    const std::vector<Scalar>& values,
                    DenseTensor* out) {
  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  DenseTensor value_tensor = Empty<T>(dev_ctx, shape);
  paddle::framework::TensorFromVector(assgin_values, dev_ctx, &value_tensor);
  value_tensor.Resize(phi::make_ddim(shape));

  SetTensorValueKernel<T, Context>(dev_ctx,
                                   x,
                                   value_tensor,
                                   starts,
                                   ends,
                                   steps,
                                   axes,
                                   decrease_axes,
                                   none_axes,
                                   out);
}

}  // namespace phi
