// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class WhereKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");

    const bool* cond_data = condition->data<bool>();
    const T* x_data = X->data<T>();
    const T* y_data = Y->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    auto x_numel = X->numel();
    for (int i = 0; i < x_numel; i++) {
      out_data[i] = cond_data[i] ? x_data[i] : y_data[i];
    }
  }
};

template <typename DeviceContext, typename T>
class WhereGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::LoDTensor>("Condition");
    const auto* cond_data = condition->data<bool>();
    auto numel = condition->numel();

    auto* dout_t =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx_t = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy_t = context.Output<framework::Tensor>(framework::GradVarName("Y"));

    auto* dout = dout_t->data<T>();
    if (dx_t != nullptr) {
      auto* dx = dx_t->mutable_data<T>(context.GetPlace());
      for (int i = 0; i < numel; i++) {
        dx[i] = dout[i] * (cond_data[i] ? 1. : 0.);
      }
    }
    if (dy_t != nullptr) {
      auto* dy = dy_t->mutable_data<T>(context.GetPlace());
      for (int i = 0; i < numel; i++) {
        dy[i] = dout[i] * (cond_data[i] ? 0. : 1.);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
