/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class SoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Output<Tensor>("Y");

    // allocate memory on device.
    Y->mutable_data<T>(context.GetPlace());

    math::SoftmaxFunctor<Place, T>()(context.device_context(), X, Y);
  }
};

template <typename Place, typename T>
class SoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Y = context.Input<Tensor>("Y");
    auto* dY = context.Input<Tensor>(framework::GradVarName("Y"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    math::SoftmaxGradFunctor<Place, T>()(context.device_context(), Y, dY, dX);
  }
};

}  // namespace operators
}  // namespace paddle
