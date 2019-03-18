/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/strided_memcpy.h"

namespace paddle {
namespace operators {  // Internal

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::Tensor;

static std::vector<int> GetOffsets(const framework::ExecutionContext& ctx) {
  std::vector<int> res;
  int rank = ctx.Input<Tensor>("X")->dims().size();
  if (ctx.HasInput("Offsets")) {
    PADDLE_ENFORCE(ctx.Attr<std::vector<int>>("offsets").empty(),
                   "Input 'Offsets' and attribute 'offsets' should not be used "
                   "at the same time.");
    const auto* offsets_tensor = ctx.Input<Tensor>("Offsets");
    PADDLE_ENFORCE_EQ(offsets_tensor->dims().size(), 1);
    PADDLE_ENFORCE_EQ(
        rank, offsets_tensor->dims()[0],
        "Offsets size should be equal to dimension size of input tensor.");
    const int* offsets_data;
    framework::Tensor cpu_tmp_tensor;
    if (platform::is_cpu_place(offsets_tensor->place())) {
      offsets_data = offsets_tensor->data<int>();
    } else {
      framework::TensorCopySync(*offsets_tensor, platform::CPUPlace(),
                                &cpu_tmp_tensor);
      offsets_data = cpu_tmp_tensor.data<int>();
    }
    res = std::vector<int>(offsets_data, offsets_data + rank);
  } else {
    res = ctx.Attr<std::vector<int>>("offsets");
    PADDLE_ENFORCE_EQ(
        rank, static_cast<int>(res.size()),
        "Offsets size should be equal to dimension size of input tensor.");
  }
  return res;
}

template <typename DeviceContext, typename T, size_t D>
void CropFunction(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* out = context.Output<Tensor>("Out");
  auto out_dims = out->dims();
  if (out_dims[0] == -1) {
    out_dims[0] = x->dims()[0];
  }
  out->mutable_data<T>(out_dims, context.GetPlace());
  auto x_stride = framework::stride(x->dims());
  auto offsets = GetOffsets(context);
  int64_t offset = 0;
  for (size_t i = 0; i < offsets.size(); ++i) {
    offset += (x_stride[i] * offsets[i]);
  }

  auto x_tensor = EigenTensor<T, D>::From(*x);
  auto out_tensor = EigenTensor<T, D>::From(*out);
  Eigen::array<int, D> e_offsets;
  Eigen::array<int, D> e_shape;
  for (size_t i = 0; i < D; ++i) {
    e_offsets[i] = offsets[i];
    e_shape[i] = out->dims()[i];
  }
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  out_tensor.device(place) = x_tensor.slice(e_offsets, e_shape);
}

template <typename DeviceContext, typename T>
class CropKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    int rank = context.Input<Tensor>("X")->dims().size();
    switch (rank) {
      case 1:
        CropFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        CropFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        CropFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        CropFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        CropFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        CropFunction<DeviceContext, T, 6>(context);
        break;
      default:
        PADDLE_THROW(
            "CropOp only support tensors with no more than 6 dimensions.");
    }
  }
};

template <typename DeviceContext, typename T, size_t D>
void CropGradFunction(const framework::ExecutionContext& context) {
  auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
  auto* x = context.Input<Tensor>("X");
  if (d_x != nullptr) {
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    d_x->mutable_data<T>(x->dims(), context.GetPlace());
    auto offsets = GetOffsets(context);
    Eigen::array<std::pair<int, int>, D> paddings;
    for (size_t i = 0; i < D; ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = d_x->dims()[i] - d_out->dims()[i] - offsets[i];
    }
    auto d_x_tensor = EigenTensor<T, D>::From(*d_x);
    auto d_out_tensor = EigenTensor<T, D>::From(*d_out);
    d_x_tensor.device(
        *context.template device_context<DeviceContext>().eigen_device()) =
        d_out_tensor.pad(paddings, 0);
  }
}

template <typename DeviceContext, typename T>
class CropGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    size_t rank =
        context.Input<Tensor>(framework::GradVarName("Out"))->dims().size();
    switch (rank) {
      case 1:
        CropGradFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        CropGradFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        CropGradFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        CropGradFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        CropGradFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        CropGradFunction<DeviceContext, T, 6>(context);
        break;
      default:
        PADDLE_THROW(
            "CropOp only support tensors with no more than 6 dimensions.");
    }
  }
};

}  // namespace operators
}  // namespace paddle
