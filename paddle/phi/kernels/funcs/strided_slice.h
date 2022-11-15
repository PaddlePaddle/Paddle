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
#include <utility>
#include <vector>

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {
static void StridedSliceOutDims(const std::vector<int64_t>& starts,
                                const std::vector<int64_t>& ends,
                                const std::vector<int64_t>& strides,
                                const std::vector<int>& axes,
                                const std::vector<int>& infer_flags,
                                const DDim in_dims,
                                const std::vector<int>& decrease_axis,
                                int64_t* out_dims_vector,
                                const size_t size,
                                bool infer_shape) {
  for (int i = 0; i < in_dims.size(); i++) {
    out_dims_vector[i] = in_dims[i];
  }
  int64_t stride_index, start_index, end_index;
  for (size_t i = 0; i < size; i++) {
    int axes_index = axes[i];
    start_index = starts[i];
    end_index = ends[i];
    stride_index = strides[i];
    bool decrease_axis_affect = false;
    if (start_index == -1 && end_index == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    if (decrease_axis_affect) {
      out_dims_vector[axes_index] = 1;
      continue;
    }
    if (infer_shape && infer_flags[i] == -1) {
      out_dims_vector[axes_index] = -1;
      continue;
    }

    PADDLE_ENFORCE_NE(
        stride_index,
        0,
        errors::InvalidArgument("stride index in StridedSlice operator is 0."));
    int64_t axis_size = in_dims[axes_index];

    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
      start_index = std::max<int64_t>(start_index, 0);
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {  // skip None stop condition
        end_index = end_index + axis_size;
        if (end_index < 0) {
          end_index = 0;
        }
      }
    }

    if (stride_index < 0) {
      start_index = start_index + 1;
      end_index = end_index + 1;
    }

    bool neg_dim_condition = ((stride_index < 0 && (start_index < end_index)) ||
                              (stride_index > 0 && (start_index > end_index)));
    PADDLE_ENFORCE_EQ(neg_dim_condition,
                      false,
                      errors::InvalidArgument(
                          "The start index and end index are invalid for their "
                          "corresponding stride."));

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, std::max(start_index, end_index));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
}

static void StridedSliceFunctor(int64_t* starts,
                                int64_t* ends,
                                int64_t* strides,
                                const int* axes,
                                int* reverse_axis,
                                const DDim dims,
                                const std::vector<int>& infer_flags,
                                const std::vector<int>& decrease_axis,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }
    bool decrease_axis_affect = false;
    if (starts[axis_index] == -1 && ends[axis_index] == 0 &&
        infer_flags[axis_index] == -1) {
      auto ret = std::find(
          decrease_axis.begin(), decrease_axis.end(), axes[axis_index]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
      starts[axis_index] = std::max<int64_t>(starts[axis_index], 0);
    }
    if (ends[axis_index] < 0) {
      if (!(ends[axis_index] == -1 &&
            strides[axis_index] < 0)) {  // skip None stop condition
        ends[axis_index] = ends[axis_index] + axis_size;
        if (ends[axis_index] < 0) {
          ends[axis_index] = 0;
        }
      }
    }
    if (decrease_axis_affect) {
      if (strides[axis_index] < 0) {
        ends[axis_index] = starts[axis_index] - 1;
      } else {
        ends[axis_index] = starts[axis_index] + 1;
      }
    }

    if (strides[axis_index] < 0) {
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        auto end_dim = axis_size - 1 < starts[axis_index] ? axis_size - 1
                                                          : starts[axis_index];
        auto offset = (end_dim - ends[axis_index]) % strides[axis_index];
        offset = offset == 0 ? strides[axis_index] : offset;

        starts[axis_index] = starts[axis_index] + offset;
        ends[axis_index] = ends[axis_index] + offset;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

template <typename Context, typename T, size_t D>
void StridedSliceCompute(const Context& dev_ctx,
                         const DenseTensor& x,
                         const std::vector<int>& axes,
                         const IntArray& starts,
                         const IntArray& ends,
                         const IntArray& strides,
                         const std::vector<int>& infer_flags,
                         const std::vector<int>& decrease_axis,
                         DenseTensor* out) {
  auto& place = *dev_ctx.eigen_device();
  DDim in_dims = x.dims();

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto reverse_axis = Eigen::array<bool, D>();

  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  StridedSliceOutDims(starts_,
                      ends_,
                      strides_,
                      axes,
                      infer_flags,
                      in_dims,
                      decrease_axis,
                      out_dims_vector.data(),
                      axes.size(),
                      false);
  DDim out_dims(phi::make_ddim(out_dims_vector));

  std::vector<int> reverse_vector(starts_.size(), 0);
  StridedSliceFunctor(starts_.data(),
                      ends_.data(),
                      strides_.data(),
                      axes.data(),
                      reverse_vector.data(),
                      in_dims,
                      infer_flags,
                      decrease_axis,
                      starts_.size());

  for (size_t axis = 0; axis < D; axis++) {
    starts_indices[axis] = 0;
    ends_indices[axis] = out_dims[axis];
    strides_indices[axis] = 1;
    reverse_axis[axis] = false;
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts_[axis];
    ends_indices[axis_index] = ends_[axis];
    strides_indices[axis_index] = strides_[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  out->Resize(out_dims);
  dev_ctx.template Alloc<T>(out);
  auto in_t = EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(x);
  auto out_t = EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
      *out, out_dims);
  if (need_reverse) {
    DenseTensor tmp;
    tmp.Resize(out_dims);
    dev_ctx.template Alloc<T>(&tmp);

    auto tmp_t =
        EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(tmp);
    tmp_t.device(place) =
        in_t.stridedSlice(starts_indices, ends_indices, strides_indices);
    out_t.device(place) = tmp_t.reverse(reverse_axis);
  } else {
    out_t.device(place) =
        in_t.stridedSlice(starts_indices, ends_indices, strides_indices);
  }

  if (decrease_axis.size() > 0) {
    out->Resize(out_dims_origin);
  }
}

template <typename Context, typename T, size_t D>
void StridedSliceCompute(const Context& dev_ctx,
                         const TensorArray& x,
                         const std::vector<int>& axes,
                         const IntArray& starts,
                         const IntArray& ends,
                         const IntArray& strides,
                         const std::vector<int>& infer_flags,
                         const std::vector<int>& decrease_axis,
                         TensorArray* out) {
  const int64_t size = x.size();
  auto in_dims = phi::make_ddim({size});

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto reverse_axis = Eigen::array<bool, D>();

  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  StridedSliceOutDims(starts_,
                      ends_,
                      strides_,
                      axes,
                      infer_flags,
                      in_dims,
                      decrease_axis,
                      out_dims_vector.data(),
                      axes.size(),
                      false);
  DDim out_dims(phi::make_ddim(out_dims_vector));

  std::vector<int> reverse_vector(starts_.size(), 0);
  StridedSliceFunctor(starts_.data(),
                      ends_.data(),
                      strides_.data(),
                      axes.data(),
                      reverse_vector.data(),
                      in_dims,
                      infer_flags,
                      decrease_axis,
                      starts_.size());

  for (size_t axis = 0; axis < D; axis++) {
    starts_indices[axis] = 0;
    ends_indices[axis] = out_dims[axis];
    strides_indices[axis] = 1;
    reverse_axis[axis] = false;
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts_[axis];
    ends_indices[axis_index] = ends_[axis];
    strides_indices[axis_index] = strides_[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  PADDLE_ENFORCE_EQ(
      starts_indices.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of start index  should be 1, but received %d.",
          starts_indices.size()));

  PADDLE_ENFORCE_EQ(
      ends_indices.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of end index should be 1, but received %d.",
          ends_indices.size()));

  PADDLE_ENFORCE_EQ(
      strides_indices.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of stride should be 1, but received %d.",
          strides_indices.size()));

  PADDLE_ENFORCE_EQ(
      out_dims_origin.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of Output should be 1, but received %d",
          out_dims_origin.size()));

  out->resize(out_dims_origin[0]);
  size_t const in_array_size = x.size();
  for (size_t i = 0; i < out->size(); i++) {
    size_t in_offset =
        (starts_indices[0] % in_array_size) + i * strides_indices[0];

    int64_t out_offset = i;
    if (need_reverse) {
      out_offset = out->size() - i - 1;
    }

    auto& in_tensor = x.at(in_offset);
    PADDLE_ENFORCE_GT(
        in_tensor.memory_size(),
        0,
        errors::PreconditionNotMet(
            "The input LoDTensorArray Input[%d] holds no memory.", in_offset));
    auto& out_tensor = out->at(out_offset);
    out_tensor.Resize(in_tensor.dims());

    phi::Copy<Context>(
        dev_ctx, in_tensor, dev_ctx.GetPlace(), false, &out_tensor);
    out_tensor.set_lod(in_tensor.lod());
  }
}

template <typename Context, typename T, size_t D>
void StridedSliceGradCompute(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& out_grad,
                             const std::vector<int>& axes,
                             const IntArray& starts,
                             const IntArray& ends,
                             const IntArray& strides,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             DenseTensor* x_grad) {
  auto& place = *dev_ctx.eigen_device();
  DDim out_dims = x.dims();

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

  auto reverse_axis = Eigen::array<bool, D>();
  std::vector<int> reverse_vector(starts_.size(), 0);

  StridedSliceFunctor(starts_.data(),
                      ends_.data(),
                      strides_.data(),
                      axes.data(),
                      reverse_vector.data(),
                      out_dims,
                      infer_flags,
                      decrease_axis,
                      starts_.size());

  for (size_t axis = 0; axis < D; axis++) {
    starts_indices[axis] = 0;
    ends_indices[axis] = out_dims[axis];
    strides_indices[axis] = 1;
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts_[axis];
    ends_indices[axis_index] = ends_[axis];
    strides_indices[axis_index] = strides_[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  dev_ctx.template Alloc<T>(x_grad);
  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, x_grad, static_cast<T>(0));

  auto out_grad_dims = out_grad.dims();

  auto in_t =
      EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(out_grad);
  auto out_t = EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
      *x_grad, out_dims);
  if (need_reverse) {
    DenseTensor reverse_input;
    reverse_input.Resize(out_grad_dims);
    dev_ctx.template Alloc<T>(&reverse_input);

    auto reverse_in_t =
        EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            reverse_input);

    reverse_in_t.device(place) = in_t.reverse(reverse_axis);
    out_t.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(place) = reverse_in_t;
  } else {
    out_t.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(place) = in_t;
  }
}

template <typename Context, typename T, size_t D>
void StridedSliceGradCompute(const Context& dev_ctx,
                             const TensorArray& x,
                             const TensorArray& out_grad,
                             const std::vector<int>& axes,
                             const IntArray& starts,
                             const IntArray& ends,
                             const IntArray& strides,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             TensorArray* x_grad) {
  // Note(weixin):Since the shape of `framework::GradVarName("Input")` of
  // StridedSliceGrad cannot be calculated by
  // `framework::GradVarName("Output")`, the dim of "Input" is used to
  // calculate the output shape. when set it to inplace OP, there may be
  // some problems.
  const int64_t size = x.size();
  DDim out_dims = phi::make_ddim({size});

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
  auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

  auto reverse_axis = Eigen::array<bool, D>();
  std::vector<int> reverse_vector(starts_.size(), 0);

  StridedSliceFunctor(starts_.data(),
                      ends_.data(),
                      strides_.data(),
                      axes.data(),
                      reverse_vector.data(),
                      out_dims,
                      infer_flags,
                      decrease_axis,
                      starts_.size());

  for (size_t axis = 0; axis < D; axis++) {
    starts_indices[axis] = 0;
    ends_indices[axis] = out_dims[axis];
    strides_indices[axis] = 1;
  }
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts_[axis];
    ends_indices[axis_index] = ends_[axis];
    strides_indices[axis_index] = strides_[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      starts_indices.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_grad_op' is `TensorArray`, the "
          "dimension of start index  should be 1, but received %d.",
          starts_indices.size()));
  PADDLE_ENFORCE_EQ(
      ends_indices.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_op' is `TensorArray`, the "
          "dimension of end index should be 1, but received %d.",
          ends_indices.size()));
  PADDLE_ENFORCE_EQ(
      strides_indices.size(),
      1,
      errors::InvalidArgument(
          "When the input of 'strided_slice_grad_op' is `TensorArray`, the "
          "dimension of stride should be 1, but received %d.",
          strides_indices.size()));

  PADDLE_ENFORCE_EQ(
      out_dims.size(),
      1,
      errors::InvalidArgument(
          "When the output of `strided_slice_grad_op` is `TensorArray`, "
          "the dimension of output should be 1, but received %d.",
          out_dims.size()));

  auto const d_out_array_size = x_grad->size();

  for (size_t j = 0; j < d_out_array_size; j++) {
    auto& dim = x.at(j).dims();
    auto& d_out_tensor = x_grad->at(j);

    int64_t sub = j - starts_indices[0];

    int64_t in_offset = sub / strides_indices[0];

    if (need_reverse) {
      in_offset = out_grad.size() - in_offset - 1;
    }

    if ((sub % strides_indices[0] == 0) && (0 <= in_offset) &&
        (static_cast<size_t>(in_offset) < out_grad.size())) {
      auto& in_tensor = out_grad.at(in_offset);
      PADDLE_ENFORCE_GT(
          in_tensor.memory_size(),
          0,
          errors::PreconditionNotMet(
              "The input LoDTensorArray Input[%d] holds no memory.",
              in_offset));

      phi::Copy<Context>(
          dev_ctx, in_tensor, dev_ctx.GetPlace(), false, &d_out_tensor);
      d_out_tensor.set_lod(in_tensor.lod());
    } else {
      d_out_tensor.Resize(dim);

      if (!d_out_tensor.IsInitialized()) {
        dev_ctx.template Alloc<T>(&d_out_tensor);
      }

      phi::funcs::SetConstant<Context, T> set_zero;
      set_zero(dev_ctx, &d_out_tensor, static_cast<T>(0));
    }
  }
}

}  // namespace funcs
}  // namespace phi
