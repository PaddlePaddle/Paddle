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

template <typename Place, typename T, typename Functor>
class ActivationKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Output<framework::Tensor>("Y");
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();
    Functor functor;
    functor(place, x, y);
  }
};

template <typename Place, typename T, typename Functor>
class ActivationGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Input<framework::Tensor>("Y");
    auto* dY = context.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    dX->mutable_data<T>(context.GetPlace());

    auto dy = framework::EigenVector<T>::Flatten(*dY);
    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();
    Functor functor;
    functor(place, x, y, dy, dx);
  }
};

struct SigmoidFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = 1. / (1. + (-x).exp());
  }
};

struct SigmoidGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * y * (1. - y);
  }
};

struct ExpFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.exp();
  }
};

struct ExpGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = y;
  }
};

template <typename T>
struct ReluFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.cwiseMax(static_cast<T>(0));
  }
};

template <typename T>
struct ReluGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * (x > static_cast<T>(0)).template cast<T>();
  }
};

}  // namespace operators
}  // namespace paddle
