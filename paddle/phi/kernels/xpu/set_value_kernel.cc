// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/set_value_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
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
                  const T* value_data,
                  const DDim& value_dims,
                  const IntArray& starts,
                  const IntArray& ends,
                  const IntArray& steps,
                  const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& decrease_axes,
                  const std::vector<int64_t>& none_axes,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto in_dims = in.dims();

  auto new_value_dims = value_dims;

  // support for 0-d tensor
  if (value_dims.size() == 0) {
    new_value_dims = {1};
  }

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
  int r = XPU_SUCCESS;
  out->Resize(in.dims());
  dev_ctx.template Alloc<T>(out);
  r = xpu::copy(dev_ctx.x_context(),
                reinterpret_cast<const XPUType*>(in.data<T>()),
                reinterpret_cast<XPUType*>(out->data<T>()),
                in.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int64_t slice_numels = phi::product(slice_dims);
  XPUType* slice_data = RAII_GUARD.alloc_l3_or_gm<XPUType>(slice_numels);

  int in_size = in_dims.size();
  std::vector<int> starts_indices(in_size, 0);
  std::vector<int> ends_indices(in_size, 0);
  std::vector<int> strides_indices(in_size, 0);
  std::vector<int> flip_axis;

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

  // Because strided_slice_view_update does not support the case of stride < 0
  // temporarily, the coordinates of starts_indices, ends_indices
  // and strides_indices need to be converted.
  // This logic may be deleted in the future.
  bool need_flip = false;
  for (size_t i = 0; i < RANK; ++i) {
    if (strides_indices[i] < 0) {
      if (!need_flip) {
        need_flip = true;
      }
      flip_axis.push_back(i);
      strides_indices[i] = strides_indices[i] * (-1);
      ends_indices[i] = starts_indices[i] + 1;
      starts_indices[i] =
          starts_indices[i] - (slice_dims[i] - 1) * strides_indices[i];
    }
  }

  auto out_shape = phi::vectorize<int>(out->dims());
  auto slice_shape = phi::vectorize<int>(slice_dims);

  // Step 2: Set slice tensor

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

  CheckIsDimsMatch(slice_dims_for_assign, new_value_dims);
  auto slice_dims_for_assign_vec = vectorize<int64_t>(slice_dims_for_assign);
  auto value_dims_vec = vectorize<int64_t>(new_value_dims);
  size_t max_dims =
      std::max(slice_dims_for_assign.size(), new_value_dims.size());
  std::vector<int64_t> ext_slice_dims_for_assign(max_dims, 1);
  std::vector<int64_t> ext_value_dims_vec(max_dims, 1);
  std::copy(slice_dims_for_assign_vec.begin(),
            slice_dims_for_assign_vec.end(),
            ext_slice_dims_for_assign.begin() +
                (max_dims - slice_dims_for_assign.size()));
  std::copy(value_dims_vec.begin(),
            value_dims_vec.end(),
            ext_value_dims_vec.begin() + (max_dims - value_dims.size()));

  r = xpu::broadcast<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(value_data),
                              slice_data,
                              ext_value_dims_vec,
                              ext_slice_dims_for_assign);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast");

  // - Step 2.2 If stride < 0, flip the slice_tensor.
  if (need_flip) {
    r = xpu::flip(dev_ctx.x_context(),
                  reinterpret_cast<const XPUType*>(slice_data),
                  slice_data,
                  slice_shape,
                  flip_axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "flip");
  }
  // Step 3: Set out tensor with value
  r = xpu::strided_slice_view_update(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(slice_data),
      reinterpret_cast<XPUType*>(out->data<T>()),
      slice_shape,
      out_shape,
      starts_indices,
      ends_indices,
      strides_indices);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_slice_view_update");
}

template <typename T, typename Context>
void SetValueKernelImpl(const Context& dev_ctx,
                        const DenseTensor& x,
                        const T* value_data,
                        const DDim& value_dims,
                        const IntArray& starts,
                        const IntArray& ends,
                        const IntArray& steps,
                        const std::vector<int64_t>& axes,
                        const std::vector<int64_t>& decrease_axes,
                        const std::vector<int64_t>& none_axes,
                        DenseTensor* out) {
  // rank是xtensor的维度信息
  const int rank = x.dims().size();

  switch (rank) {
    case 1:
      SetValueImpl<T, Context, 1>(dev_ctx,
                                  x,
                                  value_data,
                                  value_dims,
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
                                  value_data,
                                  value_dims,
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
                                  value_data,
                                  value_dims,
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
                                  value_data,
                                  value_dims,
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
                                  value_data,
                                  value_dims,
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
                                  value_data,
                                  value_dims,
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
  SetValueKernelImpl<T, Context>(dev_ctx,
                                 x,
                                 value.data<T>(),
                                 value.dims(),
                                 starts,
                                 ends,
                                 steps,
                                 axes,
                                 decrease_axes,
                                 none_axes,
                                 out);
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
  std::vector<T> assign_values;
  assign_values.reserve(values.size());
  for (const auto& val : values) {
    assign_values.push_back(val.to<T>());
  }

  auto value_dims = phi::make_ddim(shape);

  DenseTensor value_tensor;
  TensorFromVector<T>(assign_values, dev_ctx, &value_tensor);

  SetValueKernelImpl<T, Context>(dev_ctx,
                                 x,
                                 value_tensor.data<T>(),
                                 value_dims,
                                 starts,
                                 ends,
                                 steps,
                                 axes,
                                 decrease_axes,
                                 none_axes,
                                 out);
}

}  // namespace phi

PD_REGISTER_KERNEL(set_value,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetValueKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(set_value_with_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::SetTensorValueKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t) {}
