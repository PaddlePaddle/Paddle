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

#include <utility>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
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

inline framework::DDim GetSliceDims(const framework::DDim in_dims,
                                    const std::vector<int64_t> axes,
                                    const std::vector<int64_t> starts,
                                    const std::vector<int64_t> ends) {
  framework::DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    int64_t dim_value = in_dims[axis];

    int64_t start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
    int64_t end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
    start = std::max(start, static_cast<int64_t>(0));
    end = std::min(end, dim_value);

    PADDLE_ENFORCE_GT(end, start, platform::errors::InvalidArgument(
                                      "end should greater than start, but "
                                      "received end = %d, start = %d",
                                      end, start));
    slice_dims[axis] = end - start;
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
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto dtype =
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype"));
    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto* value_tensor = ctx.Input<framework::LoDTensor>("ValueTensor");

    auto in_dims = in->dims();
    auto value_dims = framework::make_ddim(shape);
    auto slice_dims = GetSliceDims(in_dims, axes, starts, ends);

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
    // - Step 1.1 Get a slice tensor from out
    Eigen::array<int64_t, D> offsets, extents;
    Eigen::array<std::pair<int64_t, int64_t>, D> paddings;

    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = slice_dims[i];
    }
    int64_t start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i] < 0 ? (starts[i] + in_dims[axes[i]]) : starts[i];
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axes[i]] = start;
    }
    for (size_t i = 0; i < paddings.size(); ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = (in_dims[i] - slice_dims[i]) - offsets[i];
    }

    slice_e.device(eigen_place) = out_e.slice(offsets, extents);

    // - Step 1.2 Get paded tensor by padding 0 to slice tensor
    pad_e.device(eigen_place) = slice_e.pad(paddings, T(0));

    // - Step 1.3 Set 0 at `_index` of out tensor
    out_e.device(eigen_place) = out_e - pad_e;

    // Step 2: Set a tensor with the same shape as out tensor. And its data at
    // '_index' is the same as value_tensor, and data out of '_index' to zero

    // - Step 2.1 Set the data of slice tensor to 0
    slice_e.device(eigen_place) = slice_e.constant(T(0));

    // - Step 2.2 Set slice tensor with value
    if (value_tensor != nullptr) {
      // ElementwiseComputeEx can do broadcasting
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_t, value_tensor, -1, SubFunctor<T>(), &slice_t);
    } else {
      Tensor value_t(dtype);
      value_t.mutable_data<T>(value_dims, place);
      auto value_name = GetValueName(dtype);
      CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
      value_t.Resize(value_dims);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_t, &value_t, -1, SubFunctor<T>(), &slice_t);
    }

    // - Step 2.3 Pad slice tensor with 0
    pad_e.device(eigen_place) = slice_e.pad(paddings, T(0));

    // Step 3: Set out tensor with value_tensor
    out_e.device(eigen_place) = out_e - pad_e;
  }
};

}  // namespace operators
}  // namespace paddle
