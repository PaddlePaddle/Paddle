/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>  // for max
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class BCELossOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* out = ctx.Output<Tensor>("Out");

    auto x_data = x->data<T>();
    auto label_data = labels->data<T>();
    auto out_data = out->mutable_data<T>(ctx.GetPlace());
    int x_numel = x->numel();

    // out = -(label * ln(x) + (1 - label) * ln(1 - x)) = (label - 1) * ln(1 -
    // x) - label * ln(x)
    for (int i = 0; i < x_numel; ++i) {
      PADDLE_ENFORCE_GE(
          x_data[i], static_cast<T>(0),
          platform::errors::InvalidArgument(
              "Illegal input, input must be greater than  or equal to 0"));
      PADDLE_ENFORCE_LE(
          x_data[i], static_cast<T>(1),
          platform::errors::InvalidArgument(
              "Illegal input, input must be less than or equal to 1"));
      out_data[i] =
          (label_data[i] - static_cast<T>(1)) *
              std::max(real_log(static_cast<T>(1) - x_data[i]), (T)(-100)) -
          label_data[i] * std::max(real_log(x_data[i]), (T)(-100));
    }
  }
};

template <typename DeviceContext, typename T>
class BCELossGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dx_data = dx->mutable_data<T>(ctx.GetPlace());
    auto dout_data = dout->data<T>();
    auto x_data = x->data<T>();
    auto label_data = labels->data<T>();

    int x_numel = x->numel();

    // dx = dout * ((x - label)/(x - x^2))
    for (int i = 0; i < x_numel; ++i) {
      dx_data[i] =
          dout_data[i] * ((x_data[i] - label_data[i]) /
                          std::max((static_cast<T>(1) - x_data[i]) * x_data[i],
                                   static_cast<T>(1e-12)));
    }
  }
};

}  // namespace operators
}  // namespace paddle
