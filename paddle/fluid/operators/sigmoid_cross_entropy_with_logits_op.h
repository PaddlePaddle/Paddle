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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

// Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    const framework::Tensor *Labels = context.Input<framework::Tensor>("Label");
    framework::Tensor *Out = context.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto labels = framework::EigenVector<T>::Flatten(*Labels);
    auto out = framework::EigenVector<T>::Flatten(*Out);
    auto &place = *context.device_context<DeviceContext>().eigen_device();

    // term1 = max(x, 0)
    auto term1 = x.cwiseMax(static_cast<T>(0));
    // term2 = x * labels
    auto term2 = x * labels;
    // term3 = log(1 + exp(-abs(x)))
    auto term3 = (static_cast<T>(1) + (-(x.abs())).exp()).log();

    out.device(place) = term1 - term2 + term3;
  }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    const framework::Tensor *Labels = context.Input<framework::Tensor>("Label");
    const framework::Tensor *dOut =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor *dX =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto labels = framework::EigenVector<T>::Flatten(*Labels);
    auto dout = framework::EigenVector<T>::Flatten(*dOut);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto sigmoid_x = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
    dx.device(place) = dout * (sigmoid_x - labels);
  }
};

}  // namespace operators
}  // namespace paddle
