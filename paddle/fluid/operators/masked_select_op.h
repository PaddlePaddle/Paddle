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
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MaskedSelectOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("input");
    auto* mask = ctx.Input<framework::Tensor>("mask") auto* output =
        ctx.Output<framework::Tensor>("Out");

    input.flatten_to_1d(input->numel());
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    auto input_dims = input->dims();

    int j = 0;
    for (sized_t i = 0; i < input->numel(); i++) {
      if (mask[i] == 1) {
        out_data[j] = input_data[i];
        j++;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class MaskedSelectGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* output_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("input"));
    auto* mask = ctx.Input<framework::Tensor>("mask") int j = 0;
    mask.flatten_to_1d(mask->numel());
    const* mask_data = mask->data<bool>();

    for (sized_t i = 0; i < mask->numel(); i++) {
      if (mask_data[i]) {
        input_grad[i] == output_grad[j];
        j++;
      } else {
        input_grad[i] == 0;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
