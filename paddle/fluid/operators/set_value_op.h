//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline std::string GetValueName(framework::proto::VarType::Type data_type) {
  std::string value_name;
  switch (data_type) {
    case framework::proto::VarType::INT32:
      value_name = "int32_values";
      break;
    case framework::proto::VarType::INT64:
      value_name = "int64_values";
      break;
    case framework::proto::VarType::FP32:
      value_name = "fp32_values";
      break;
    case framework::proto::VarType::FP64:
      value_name = "fp64_values";
      break;
    case framework::proto::VarType::BOOL:
      value_name = "bool_values";
      break;

    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type(code %d) for SetValue operator, only "
          "supports bool, int32, float32 and int64.",
          data_type));
  }
  return value_name;
}

inline void CheckAndUpdateSlice(const framework::DDim in_dims,
                                const std::vector<int64_t> axes,
                                std::vector<int64_t>* starts,
                                std::vector<int64_t>* ends,
                                std::vector<int64_t>* steps) {
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    int64_t dim_value = in_dims[axis];

    int64_t start =
        (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
    int64_t end = (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
    start = std::max(start, static_cast<int64_t>(0));
    end = std::min(end, dim_value);

    int64_t step = (*steps)[i];
    PADDLE_ENFORCE_NE(
        step, 0, platform::errors::InvalidArgument(
                     "Step should not be 0, but received step = %d.", step));
    if (step > 0) {
      start = std::min(start, dim_value);
      end = std::max(end, static_cast<int64_t>(0));
      PADDLE_ENFORCE_GT(
          end, start,
          platform::errors::InvalidArgument(
              "When step > 0, end should be greater than start, but "
              "received end = %d, start = %d.",
              end, start));
    } else {
      // NOTE(liym27): When step < 0, start should less and equal to dim_value-1
      // "end is -1" means contain the 0-th element of this axis.
      start = std::min(start, dim_value - 1);
      end = std::max(end, static_cast<int64_t>(-1));
      PADDLE_ENFORCE_GT(
          start, end,
          platform::errors::InvalidArgument(
              "When step < 0, start should be greater than end, but "
              "received start = %d, end = %d.",
              start, end));
    }

    (*starts)[i] = start;
    (*ends)[i] = end;
  }
}

inline framework::DDim GetSliceDims(const framework::DDim in_dims,
                                    const std::vector<int64_t> axes,
                                    const std::vector<int64_t> starts,
                                    const std::vector<int64_t> ends,
                                    const std::vector<int64_t> steps) {
  framework::DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    int64_t start = starts[i];
    int64_t end = ends[i];
    int64_t step = steps[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

template <typename DeviceContext, typename T>
class SetValueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const int rank = ctx.Input<framework::LoDTensor>("Input")->dims().size();

    // TODO(liym27): A more elegent code to do this. C++ has to make template
    //  integer as constant, but we had better have alternative writing in the
    //  future.
    switch (rank) {
      case 1:
        SetValueCompute<1>(ctx);
        break;
      case 2:
        SetValueCompute<2>(ctx);
        break;
      case 3:
        SetValueCompute<3>(ctx);
        break;
      case 4:
        SetValueCompute<4>(ctx);
        break;
      case 5:
        SetValueCompute<5>(ctx);
        break;
      case 6:
        SetValueCompute<6>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input should be less than 7, but received %d.", rank));
    }
  }

 private:
  template <size_t D>
  void SetValueCompute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::LoDTensor>("Input");
    auto* value_tensor = ctx.Input<framework::LoDTensor>("ValueTensor");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto starts_tensor_list =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto ends_tensor_list = ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto steps_tensor_list =
        ctx.MultiInput<framework::Tensor>("StepsTensorList");

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto steps = ctx.Attr<std::vector<int64_t>>("steps");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");

    auto dtype = in->type();
    if (!starts_tensor_list.empty()) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }
    if (!ends_tensor_list.empty()) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }
    if (!steps_tensor_list.empty()) {
      steps = GetDataFromTensorList<int64_t>(steps_tensor_list);
    }

    auto in_dims = in->dims();
    CheckAndUpdateSlice(in_dims, axes, &starts, &ends, &steps);
    auto slice_dims = GetSliceDims(in_dims, axes, starts, ends, steps);

    auto place = ctx.GetPlace();
    auto& eigen_place =
        *ctx.template device_context<DeviceContext>().eigen_device();

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
    TensorCopy(*in, place, out);

    Tensor slice_t(dtype), pad_t(dtype);
    slice_t.mutable_data<T>(slice_dims, place);
    pad_t.mutable_data<T>(in_dims, place);

    auto pad_e = framework::EigenTensor<T, D>::From(pad_t, in_dims);
    auto out_e = framework::EigenTensor<T, D>::From(*out);
    auto slice_e = framework::EigenTensor<T, D>::From(slice_t, slice_dims);

    // Step 1: Set the value of out at `_index` to zero
    slice_e.device(eigen_place) = slice_e.constant(T(0));

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    for (size_t i = 0; i < D; ++i) {
      starts_indices[i] = 0;
      ends_indices[i] = slice_dims[i];
      strides_indices[i] = 1;
    }
    for (size_t i = 0; i < axes.size(); i++) {
      int axis_index = axes[i];
      starts_indices[axis_index] = starts[i];
      ends_indices[axis_index] = ends[i];
      strides_indices[axis_index] = steps[i];
    }

    out_e.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(eigen_place) = slice_e;

    // Step 2: Set a tensor with the same shape as out tensor. And its data at
    // '_index' is the same as value_tensor, and data out of '_index' to zero
    // - Step 2.1 Set slice tensor with value
    if (value_tensor != nullptr) {
      // ElementwiseComputeEx can do broadcasting
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_t, value_tensor, -1, SubFunctor<T>(), &slice_t);
    } else {
      Tensor value_t(dtype);
      auto value_dims = framework::make_ddim(shape);
      value_t.mutable_data<T>(value_dims, place);
      auto value_name = GetValueName(dtype);
      CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
      value_t.Resize(value_dims);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_t, &value_t, -1, SubFunctor<T>(), &slice_t);
    }

    // - Step 2.2 Pad slice tensor with 0
    pad_e.device(eigen_place) = pad_e.constant(T(0));
    pad_e.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(eigen_place) = slice_e;

    // Step 3: Set out tensor with value_tensor
    out_e.device(eigen_place) = out_e - pad_e;
  }
};

}  // namespace operators
}  // namespace paddle
