/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/fluid/operators/math/hardmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class HardmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    math::HardmaxFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), X, Out);
  }
};

template <typename DeviceContext, typename T>
class HardmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));

    dX->mutable_data<T>(context.GetPlace());

    // allocate memory on device.
    math::HardmaxGradFunctor<DeviceContext, T>()(
        context.template device_context<DeviceContext>(), Out, dOut, dX);
  }
};

}  // namespace operators
}  // namespace paddle
