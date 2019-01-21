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
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct SigmoidCrossEntropyWithLogitsForward {
  HOSTDEVICE SigmoidCrossEntropyWithLogitsForward(const int &ignore_index)
      : ignore_index(ignore_index) {}

  HOSTDEVICE T operator()(const T &x, const T &label) const {
    if (static_cast<int>(label) == ignore_index) {
      return static_cast<T>(0.);
    }
    T term1 = (x > 0) ? x : 0;
    T term2 = x * label;
    T term3 = std::log(static_cast<T>(1) + std::exp(-(std::abs(x))));
    return term1 - term2 + term3;
  }

  int ignore_index;
};

template <typename T>
struct SigmoidCrossEntropyWithLogitsBackward {
  HOSTDEVICE SigmoidCrossEntropyWithLogitsBackward(const int &ignore_index)
      : ignore_index(ignore_index) {}

  HOSTDEVICE T operator()(const T &x, const T &label) const {
    if (static_cast<int>(label) == ignore_index) {
      return static_cast<T>(0.);
    }
    T simoid_x = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    return simoid_x - label;
  }

  int ignore_index;
};

// Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    Tensor *Out = context.Output<Tensor>("Out");
    Out->mutable_data<T>(context.GetPlace());
    int ignore_index = context.Attr<int>("ignore_index");

    auto x = EigenVector<T>::Flatten(*X);
    auto labels = EigenVector<T>::Flatten(*Labels);
    auto out = EigenVector<T>::Flatten(*Out);
    auto &place = *context.device_context<DeviceContext>().eigen_device();

    out.device(place) = x.binaryExpr(
        labels, SigmoidCrossEntropyWithLogitsForward<T>(ignore_index));
  }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    const Tensor *dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    auto ignore_index = context.Attr<int>("ignore_index");
    auto x = EigenVector<T>::Flatten(*X);
    auto labels = EigenVector<T>::Flatten(*Labels);
    auto dout = EigenVector<T>::Flatten(*dOut);
    auto dx = EigenVector<T>::Flatten(*dX);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto diff = x.binaryExpr(labels, SigmoidCrossEntropyWithLogitsBackward<T>(
                                         static_cast<int>(ignore_index)));
    dx.device(place) = dout * diff;
  }
};

}  // namespace operators
}  // namespace paddle
