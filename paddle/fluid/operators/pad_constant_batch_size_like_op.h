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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/padding.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PadConstantBatchSizeLikeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto in_x = context.Input<framework::Tensor>("X");
    auto in_y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");

    if (in_x->dims()[0] == in_y->dims()[0]) {
      //      TensorCopy(in_y, context.GetPlace(), context, out);
      out->ShareDataWith(*in_y);
      return;
    }

    T pad_value = context.Attr<T>("pad_value");
    out->mutable_data<T>(context.GetPlace());

    int rank = context.Input<framework::Tensor>("X")->dims().size();

    std::vector<int> pads(rank * 2, 0);
    pads[1] = static_cast<int>(in_x->dims()[0] - in_y->dims()[0]);

    switch (rank) {
      case 1:
        math::PadConstantBatchSizeLikeFunction<DeviceContext, T, 1>(
            context, pads, *in_y, pad_value, out);
        break;
      case 2:
        math::PadConstantBatchSizeLikeFunction<DeviceContext, T, 2>(
            context, pads, *in_y, pad_value, out);
        break;
      case 3:
        math::PadConstantBatchSizeLikeFunction<DeviceContext, T, 3>(
            context, pads, *in_y, pad_value, out);
        break;
      case 4:
        math::PadConstantBatchSizeLikeFunction<DeviceContext, T, 4>(
            context, pads, *in_y, pad_value, out);
        break;
      case 5:
        math::PadConstantBatchSizeLikeFunction<DeviceContext, T, 5>(
            context, pads, *in_y, pad_value, out);
        break;
      case 6:
        math::PadConstantBatchSizeLikeFunction<DeviceContext, T, 6>(
            context, pads, *in_y, pad_value, out);
        break;
      default:
        PADDLE_THROW(
            "PadConstantBatchSizeLikeOp only support tensors with no more than "
            "6 dimensions.");
    }
  }
};

template <typename DeviceContext, typename T>
class PadConstantBatchSizeLikeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto in_x = context.Input<framework::Tensor>("X");
    auto in_y = context.Input<framework::Tensor>("Y");
    auto in_dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_y = context.Output<framework::Tensor>(framework::GradVarName("Y"));

    if (d_y == nullptr) {
      return;
    }

    if (in_x->dims()[0] == in_y->dims()[0]) {
      // TensorCopy(in_y, context.GetPlace(), context, out);
      d_y->ShareDataWith(*in_dout);
      return;
    }

    d_y->mutable_data<T>(context.GetPlace());
    int rank = in_dout->dims().size();

    std::vector<int> pads(static_cast<size_t>(rank) * 2, 0);
    pads[1] = static_cast<int>(in_y->dims()[0] - in_x->dims()[0]);

    switch (rank) {
      case 1:
        math::PadConstantBatchSizeLikeGradFunction<DeviceContext, T, 1>(
            context, pads, *in_dout, d_y);
        break;
      case 2:
        math::PadConstantBatchSizeLikeGradFunction<DeviceContext, T, 2>(
            context, pads, *in_dout, d_y);
        break;
      case 3:
        math::PadConstantBatchSizeLikeGradFunction<DeviceContext, T, 3>(
            context, pads, *in_dout, d_y);
        break;
      case 4:
        math::PadConstantBatchSizeLikeGradFunction<DeviceContext, T, 4>(
            context, pads, *in_dout, d_y);
        break;
      case 5:
        math::PadConstantBatchSizeLikeGradFunction<DeviceContext, T, 5>(
            context, pads, *in_dout, d_y);
        break;
      case 6:
        math::PadConstantBatchSizeLikeGradFunction<DeviceContext, T, 6>(
            context, pads, *in_dout, d_y);
        break;
      default:
        PADDLE_THROW(
            "PadConstantBatchSizeLikeOp only support tensors with no more than "
            "6 dimensions.");
    }
  }
};

}  // namespace operators
}  // namespace paddle
