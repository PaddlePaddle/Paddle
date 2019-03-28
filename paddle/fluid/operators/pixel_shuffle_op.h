/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PixelShuffleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");

    auto input_dims = input->dims();
    auto num = input_dims[0];
    auto channel = input_dims[1];
    auto height = input_dims[2];
    auto width = input_dims[3];

    int count = num * channel * height * width;
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    for (int index = 0; index < count; index++) {
      output_data[index] = input_data[index];
    }
  }
};

template <typename DeviceContext, typename T>
class PixelShuffleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto input_dims = input->dims();
    auto num = input_dims[0];
    auto channel = input_dims[1];
    auto height = input_dims[2];
    auto width = input_dims[3];

    int count = num * channel * height * width;
    auto* output_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    const T* output_grad_data = output_grad->data<T>();

    for (int index = 0; index < count; index++) {
      input_grad_data[index] = output_grad_data[index];
    }
  }
};

}  // namespace operators
}  // namespace paddle
