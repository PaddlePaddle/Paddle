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

#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/strided_slice_op.h"

namespace phi {

inline void GetOffsets(const DDim& big_dim,
                       const DDim& small_dim,
                       DDim start_offset,
                       int cur_dim,
                       std::vector<DDim>* offsets) {
  if (cur_dim == big_dim.size()) {
    offsets->push_back(start_offset);
    return;
  }
  if (small_dim[cur_dim] == big_dim[cur_dim]) {
    GetOffsets(big_dim, small_dim, start_offset, cur_dim + 1, offsets);
  } else {
    for (int i = 0; i < big_dim[cur_dim]; i++) {
      GetOffsets(big_dim, small_dim, start_offset, cur_dim + 1, offsets);
      start_offset[cur_dim] += 1;
    }
  }
}

template <typename T, typename Context, size_t RANK>
void SetValueGradImpl(const Context& dev_ctx,
                      const DenseTensor& out_grad,
                      const ScalarArray& starts,
                      const ScalarArray& ends,
                      const ScalarArray& steps,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& decrease_axes,
                      const std::vector<int64_t>& none_axes,
                      DenseTensor* x_grad,
                      DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(
      out_grad.IsInitialized(),
      true,
      errors::PermissionDenied(
          "The input of `set_value_grad`(out_grad) has not been initialized"));

  auto in_dims = out_grad.dims();

  std::vector<int> decrease_axis_int32(decrease_axes.begin(),
                                       decrease_axes.end());
  std::vector<int> axes_int32(axes.begin(), axes.end());
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
  paddle::operators::StridedSliceOutDims(starts_local,
                                         ends_local,
                                         steps_local,
                                         axes_int32,
                                         infer_flags,
                                         in_dims,
                                         decrease_axis_int32,
                                         out_dims_vector.data(),
                                         axes.size(),
                                         false);

  DDim out_dims(phi::make_ddim(out_dims_vector));

  std::vector<int> reverse_vector(starts_local.size(), 0);
  paddle::operators::StridedSliceFunctor(starts_local.data(),
                                         ends_local.data(),
                                         steps_local.data(),
                                         axes_int32.data(),
                                         reverse_vector.data(),
                                         in_dims,
                                         infer_flags,
                                         decrease_axis_int32,
                                         starts_local.size());

  auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, RANK>();
  auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, RANK>();
  auto steps_indices = Eigen::DSizes<Eigen::DenseIndex, RANK>();
  auto reverse_axis = Eigen::array<bool, RANK>();

  for (size_t axis = 0; axis < RANK; axis++) {
    starts_indices[axis] = 0;
    ends_indices[axis] = out_dims[axis];
    steps_indices[axis] = 1;
    reverse_axis[axis] = false;
  }

  for (size_t axis = 0; axis < axes.size(); axis++) {
    int axis_index = axes[axis];
    starts_indices[axis_index] = starts_local[axis];
    ends_indices[axis_index] = ends_local[axis];
    steps_indices[axis_index] = steps_local[axis];
    reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
  }

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }

  auto& place = *dev_ctx.eigen_device();
  phi::funcs::SetConstant<Context, T> set_zero;

  if (x_grad) {
    // Set gradient of `Input`
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

    auto x_grad_t =
        EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(*x_grad);

    DenseTensor tmp = Full<T>(dev_ctx, out_dims_vector, static_cast<T>(0));
    auto tmp_t =
        EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(tmp);

    x_grad_t.stridedSlice(starts_indices, ends_indices, steps_indices)
        .device(place) = tmp_t;
  }
  if (value_grad) {
    dev_ctx.template Alloc<T>(value_grad);
    set_zero(dev_ctx, value_grad, static_cast<T>(0));

    auto in_t = EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(
        out_grad);

    if (value_grad->dims() == out_dims) {
      auto value_grad_t =
          EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(
              *value_grad);
      if (need_reverse) {
        DenseTensor tmp = Full<T>(dev_ctx, out_dims_vector, static_cast<T>(0));
        auto tmp_t =
            EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(tmp);

        tmp_t.device(place) =
            in_t.stridedSlice(starts_indices, ends_indices, steps_indices);
        value_grad_t.device(place) = tmp_t.reverse(reverse_axis);
      } else {
        value_grad_t.device(place) =
            in_t.stridedSlice(starts_indices, ends_indices, steps_indices);
      }
    } else {
      int out_dims_size = out_dims.size();
      auto value_grad_dims = value_grad->dims();
      auto fake_value_grad_dims = out_dims;

      // Create an extented shape according to the rules of broadcast.
      auto value_grad_dims_size = value_grad_dims.size();

      int num_decrease = 0;

      int decrease_axis_size = decrease_axes.size();
      for (int i = 0; i < out_dims_size; i++) {
        if (decrease_axes.end() !=
            std::find(decrease_axes.begin(), decrease_axes.end(), i)) {
          fake_value_grad_dims[i] = 1;
          num_decrease++;
        } else if (i < out_dims_size - (value_grad_dims_size +
                                        decrease_axis_size - num_decrease)) {
          fake_value_grad_dims[i] = 1;
        } else {
          auto index_grad =
              i - (out_dims_size -
                   (value_grad_dims_size + decrease_axis_size - num_decrease));
          fake_value_grad_dims[i] = value_grad_dims[index_grad];

          PADDLE_ENFORCE_EQ((out_dims[i] == value_grad_dims[index_grad]) ||
                                (value_grad_dims[index_grad] == 1),
                            true,
                            errors::InvalidArgument(
                                "An error occurred while calculating %s: "
                                "[%s] can not be accumulated into [%s].",
                                paddle::framework::GradVarName("ValueTensor"),
                                out_dims,
                                value_grad_dims));
        }
      }

      VLOG(3) << "Dimensions of "
              << paddle::framework::GradVarName("ValueTensor") << "(["
              << value_grad_dims << "])is broadcasted into ["
              << fake_value_grad_dims << "].";

      auto extent = Eigen::DSizes<Eigen::DenseIndex, RANK>();
      auto offset = out_dims;
      for (int i = 0; i < out_dims_size; i++) {
        offset[i] = 0;
        extent[i] = fake_value_grad_dims[i];
      }
      std::vector<DDim> offsets;
      GetOffsets(out_dims, fake_value_grad_dims, offset, 0, &offsets);

      auto value_grad_t =
          EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(
              *value_grad, fake_value_grad_dims);

      DenseTensor tmp = Full<T>(dev_ctx, out_dims_vector, static_cast<T>(0));
      auto tmp_t =
          EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(tmp);

      tmp_t.device(place) =
          in_t.stridedSlice(starts_indices, ends_indices, steps_indices);

      // accumulate gradient
      for (auto offset : offsets) {
        value_grad_t.device(place) =
            value_grad_t + tmp_t.slice(EigenDim<RANK>::From(offset), extent);
      }
      if (need_reverse) {
        DenseTensor tmp_value =
            Full<T>(dev_ctx,
                    {fake_value_grad_dims.Get(), fake_value_grad_dims.size()},
                    static_cast<T>(0));
        auto tmp_value_t =
            EigenTensor<T, RANK, Eigen::RowMajor, Eigen::DenseIndex>::From(
                tmp_value);
        tmp_value_t.device(place) = value_grad_t.reverse(reverse_axis);
        value_grad_t.device(place) = tmp_value_t;
      }
    }
  }
}

template <typename T, typename Context>
void SetValueGradKernel(const Context& dev_ctx,
                        const DenseTensor& out_grad,
                        const ScalarArray& starts,
                        const ScalarArray& ends,
                        const ScalarArray& steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        DenseTensor* x_grad,
                        DenseTensor* value_grad) {
  const int rank = out_grad.dims().size();

  switch (rank) {
    case 1:
      SetValueGradImpl<T, Context, 1>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 2:
      SetValueGradImpl<T, Context, 2>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 3:
      SetValueGradImpl<T, Context, 3>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 4:
      SetValueGradImpl<T, Context, 4>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 5:
      SetValueGradImpl<T, Context, 5>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    case 6:
      SetValueGradImpl<T, Context, 6>(dev_ctx,
                                      out_grad,
                                      starts,
                                      ends,
                                      steps,
                                      axes,
                                      decrease_axes,
                                      none_axes,
                                      x_grad,
                                      value_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of set_value_grad's input should be less than 7, but "
          "received %d.",
          rank));
  }
}

}  // namespace phi
