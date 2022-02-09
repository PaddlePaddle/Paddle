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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 6

namespace pten {

inline std::vector<int> get_expand_shape(
    const framework::ExecutionContext& ctx) {
  if (ctx.HasInput("Shape")) {
    auto* shape_tensor = ctx.Input<framework::LoDTensor>("Shape");
    auto* shape_data = shape_tensor->data<int>();
    framework::Tensor cpu_shape_tensor;
    if (platform::is_gpu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(*shape_tensor, platform::CPUPlace(),
                                        &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(*shape_tensor, platform::CPUPlace(),
                                        &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#endif
#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(shape_tensor->place())) {
      paddle::framework::TensorCopySync(*shape_tensor, platform::CPUPlace(),
                                        &cpu_shape_tensor);
      shape_data = cpu_shape_tensor.data<int>();
    }
#endif
    auto vec_shape =
        std::vector<int>(shape_data, shape_data + shape_tensor->numel());
    return vec_shape;
  }

  auto list_expand_shapes_tensor =
      ctx.MultiInput<framework::Tensor>("expand_shapes_tensor");
  if (list_expand_shapes_tensor.size() > 0) {
    // get tensor from
    std::vector<int> vec_epxand_shape;
    for (size_t i = 0; i < list_expand_shapes_tensor.size(); ++i) {
      auto tensor = list_expand_shapes_tensor[i];
      if (platform::is_gpu_place(tensor->place())) {
        framework::Tensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#ifdef PADDLE_WITH_ASCEND_CL
      else if (platform::is_npu_place(tensor->place())) {  // NOLINT
        framework::Tensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#endif
#ifdef PADDLE_WITH_XPU
      else if (platform::is_xpu_place(tensor->place())) {  // NOLINT
        framework::Tensor temp;
        paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_shape.push_back(*temp.data<int32_t>());
      }
#endif
      else {  // NOLINT
        vec_epxand_shape.push_back(*tensor->data<int32_t>());
      }
    }
    return vec_epxand_shape;
  } else {
    return ctx.Attr<std::vector<int>>("shape");
  }
}

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::To32BitIndex;

template <typename DeviceContext, typename T>
class ExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The rank of the input 'X' for expand_v2 op must be positive, "
            "but the value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The rank of the input 'X' for expand_v2 op must be less than "
            "or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, rank));
    auto expand_shape = get_expand_shape(context);
    auto shape_size = expand_shape.size();
    PADDLE_ENFORCE_GE(
        shape_size, rank,
        platform::errors::InvalidArgument(
            "The number (%d) of elements of 'shape' for expand_v2 op must be "
            "greater than or equal to the rank (%d) of the input 'X'.",
            shape_size, rank));
    PADDLE_ENFORCE_LE(
        shape_size, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The number (%d) of elements of 'shape' for expand_v2 op must be "
            "less than or equal to %d.",
            shape_size, MAX_RANK_SUPPORTED));
    rank = std::max(rank, static_cast<int>(shape_size));
    switch (rank) {
      case 1:
        Expand<1>(context);
        break;
      case 2:
        Expand<2>(context);
        break;
      case 3:
        Expand<3>(context);
        break;
      case 4:
        Expand<4>(context);
        break;
      case 5:
        Expand<5>(context);
        break;
      case 6:
        Expand<6>(context);
        break;
    }
  }

 protected:
  template <int Rank>
  void Expand(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<Tensor>("X");

    auto in_dims = in0->dims();
    auto expand_shape = get_expand_shape(context);
    auto vec_in_dims = framework::vectorize<int>(in_dims);
    auto diff = expand_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> repeat_times(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      PADDLE_ENFORCE_NE(expand_shape[i], 0,
                        platform::errors::InvalidArgument(
                            "The expanded size cannot be zero."));
      if (i < diff) {
        PADDLE_ENFORCE_GT(
            expand_shape[i], 0,
            platform::errors::InvalidArgument(
                "The expanded size (%d) for non-existing dimensions must be "
                "positive for expand_v2 op.",
                expand_shape[i]));
        repeat_times[i] = expand_shape[i];
      } else if (expand_shape[i] > 0) {
        if (vec_in_dims[i] != 1) {
          PADDLE_ENFORCE_EQ(
              vec_in_dims[i], expand_shape[i],
              platform::errors::InvalidArgument(
                  "The value (%d) of the non-singleton dimension does not match"
                  " the corresponding value (%d) in shape for expand_v2 op.",
                  vec_in_dims[i], expand_shape[i]));
          repeat_times[i] = 1;
        } else {
          repeat_times[i] = expand_shape[i];
        }
      } else {
        PADDLE_ENFORCE_EQ(
            expand_shape[i], -1,
            platform::errors::InvalidArgument(
                "When the value in shape is negative for expand_v2 op, "
                "only -1 is supported, but the value received is %d.",
                expand_shape[i]));
        repeat_times[i] = 1;
      }
    }

    auto* out0 = context.Output<Tensor>("Out");
    Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      bcast_dims[i] = repeat_times[i];
    }

    framework::DDim new_in_dims = framework::make_ddim(vec_in_dims);
    framework::DDim out_dims(new_in_dims);
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      out_dims[i] *= repeat_times[i];
    }

    out0->Resize(out_dims);
    auto x = EigenTensor<T, Rank>::From(*in0, new_in_dims);
    out0->mutable_data<T>(context.GetPlace());
    auto y = EigenTensor<T, Rank>::From(*out0, out_dims);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    // use 32-bit index to speed up
    bool use_32bit_index = y.size() < Eigen::NumTraits<int>::highest();
    if (use_32bit_index) {
      EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
          place, To32BitIndex(y), To32BitIndex(x), bcast_dims);
    } else {
      EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(place, y, x,
                                                                   bcast_dims);
    }
  }
};

}  // namespace pten