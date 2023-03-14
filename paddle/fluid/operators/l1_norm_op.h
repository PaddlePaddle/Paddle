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
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace paddle {
namespace operators {

// Out = sum(abs(X))
template <typename DeviceContext, typename T>
class L1NormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const phi::DenseTensor *X = context.Input<phi::DenseTensor>("X");
    phi::DenseTensor *Out = context.Output<phi::DenseTensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto out = framework::EigenScalar<T>::From(*Out);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    EigenL1Norm<std::decay_t<decltype(place)>, T>::Eval(place, out, x);
  }
};

// dX = dout * sign(X)
template <typename DeviceContext, typename T>
class L1NormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const phi::DenseTensor *x = context.Input<phi::DenseTensor>("X");
    const phi::DenseTensor *d_out =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(
        d_out->numel(),
        1,
        platform::errors::InvalidArgument(
            "Input(GRAD@Out) of L1NormGradOP should be a scalar."));
    phi::DenseTensor *dx =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(context.GetPlace());

    auto x_eigen = framework::EigenVector<T>::Flatten(*x);
    auto d_out_eigen = framework::EigenVector<T>::Flatten(*d_out);
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<Eigen::DenseIndex, 1> x_dsize(x->numel());
    EigenL1NormGrad<std::decay_t<decltype(place)>, T>::Eval(
        place, dx_eigen, d_out_eigen, x_eigen, x_dsize);
  }
};

}  // namespace operators
}  // namespace paddle
