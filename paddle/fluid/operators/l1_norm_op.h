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

// Out = sum(abs(X))
template <typename DeviceContext, typename T>
class L1NormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    framework::Tensor *Out = context.Output<framework::Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto out = framework::EigenScalar<T>::From(*Out);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    out.device(place) = x.abs().sum();
  }
};

// dX = dout * sign(X)
template <typename DeviceContext, typename T>
class L1NormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *x = context.Input<framework::Tensor>("X");
    const framework::Tensor *d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(
        d_out->numel(), 1,
        platform::errors::InvalidArgument(
            "Input(GRAD@Out) of L1NormGradOP should be a scalar."));
    framework::Tensor *dx =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(context.GetPlace());

    auto x_eigen = framework::EigenVector<T>::Flatten(*x);
    auto d_out_eigen = framework::EigenVector<T>::Flatten(*d_out);
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 1> x_dsize(x->numel());
    dx_eigen.device(place) = d_out_eigen.broadcast(x_dsize) * x_eigen.sign();
  }
};

}  // namespace operators
}  // namespace paddle
