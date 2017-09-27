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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// Y = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename Place, typename T>
class SigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Labels");
    Tensor *Y = context.Output<Tensor>("Y");
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto labels = framework::EigenVector<T>::Flatten(*Labels);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();

    // term1 = max(x, 0)
    auto term1 = x.cwiseMax(static_cast<T>(0));
    // term2 = x * labels
    auto term2 = x * labels;
    // term3 = log(1 + exp(-abs(x)))
    auto term3 = (static_cast<T>(1) + (-(x.abs())).exp()).log();

    y.device(place) = term1 - term2 + term3;
  }
};

// dX = sigmoid(X) - labels
template <typename Place, typename T>
class SigmoidCrossEntropyWithLogitsGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Labels");
    const Tensor *dY = context.Input<Tensor>(framework::GradVarName("Y"));
    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto labels = framework::EigenVector<T>::Flatten(*Labels);
    auto dy = framework::EigenVector<T>::Flatten(*dY);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();

    auto sigmoid_x = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
    dx.device(place) = dy * (sigmoid_x - labels);
  }
};

}  // namespace operators
}  // namespace paddle
