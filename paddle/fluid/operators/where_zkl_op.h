/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class WhereZklKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* tensor_condition = context.Input<Tensor>("Condition");
    const Tensor* tensor_x = context.Input<Tensor>("X");
    const Tensor* tensor_y = context.Input<Tensor>("Y");
    Tensor* tensor_out = context.Output<Tensor>("Out");

    const auto* data_condition = tensor_condition->data<bool>();
    const auto* data_x = tensor_x->data<T>();
    const auto* data_y = tensor_y->data<T>();
    auto* data_out =
        tensor_out->mutable_data<T>(tensor_x->dims(), context.GetPlace());

    int size = tensor_x->numel();

    for (int i = 0; i < size; ++i)
      data_out[i] = data_condition[i] ? data_x[i] : data_y[i];
  }
};

template <typename DeviceContext, typename T>
class WhereZklGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor_condition = context.Input<Tensor>("Condition");
    auto* tensor_out = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* tensor_dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* tensor_dy = context.Output<Tensor>(framework::GradVarName("Y"));

    const auto* data_condition = tensor_condition->data<bool>();
    const auto* data_out = tensor_out->data<T>();

    auto* data_dx =
        tensor_dx->mutable_data<T>(tensor_out->dims(), context.GetPlace());
    auto* data_dy =
        tensor_dy->mutable_data<T>(tensor_out->dims(), context.GetPlace());

    int size = tensor_out->numel();

    for (int i = 0; i < size; ++i) {
      data_dx[i] = data_condition[i] ? data_out[i] : 0;
      data_dy[i] = data_condition[i] ? 0 : data_out[i];
    }
  }
};

}  // namespace operators
}  // namespace paddle
