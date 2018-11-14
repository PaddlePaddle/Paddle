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

// Out = sum(square(X))
template <typename DeviceContext, typename T>
class SquaredL2NormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    framework::Tensor *Out = context.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto out = framework::EigenScalar<T>::From(*Out);
    auto *place =
        context.template device_context<DeviceContext>().eigen_device();

    out.device(*place) = x.square().sum();
  }
};

// dX = X
template <typename DeviceContext, typename T>
class SquaredL2NormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    const framework::Tensor *dOut =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE(dOut->numel() == 1,
                   "Squared L2 Norm Gradient should be scalar");
    framework::Tensor *dX =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto dout = framework::EigenVector<T>::Flatten(*dOut);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto *place =
        context.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 1> x_dsize(X->numel());
    dx.device(*place) = (dout.broadcast(x_dsize) * x) * static_cast<T>(2.0);
  }
};

}  // namespace operators
}  // namespace paddle
