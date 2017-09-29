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
class ActivationKernel : public framework::OpKernel<T> {
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
class ActivationGradKernel : public framework::OpKernel<T> {
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

// sigmoid(x) = 1 / (1 + exp(-x))
template <typename T>
struct SigmoidFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
  }
};

template <typename T>
struct SigmoidGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * y * (static_cast<T>(1) - y);
  }
};

// exp(x) = e^x
struct ExpFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.exp();
  }
};

struct ExpGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * y;
  }
};

// relu(x) = max(x, 0)
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

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
struct TanhFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.tanh();
  }
};

template <typename T>
struct TanhGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * (static_cast<T>(1) - y * y);
  }
};

// sqrt(x) = x^(1/2)
struct SqrtFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.sqrt();
  }
};

template <typename T>
struct SqrtGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    const Y y_conj = Eigen::numext::conj(y);
    dx.device(d) = static_cast<T>(0.5) * dy / y_conj;
  }
};

// abs(x) = |x|
struct AbsFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.abs();
  }
};

struct AbsGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * x.sign();
  }
};

// reciprocal(x) = 1 / x
template <typename T>
struct ReciprocalFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = static_cast<T>(1) / x;
  }
};

template <typename T>
struct ReciprocalGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * static_cast<T>(-1) * y * y;
  }
};

// log(x) = natural logarithm of x
struct LogFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.log();
  }
};

template <typename T>
struct LogGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * (static_cast<T>(1) / x);
  }
};

// square(x) = x^2
struct SquareFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x.square();
  }
};

template <typename T>
struct SquareGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) = dy * static_cast<T>(2) * x;
  }
};

// softsign(x) = x / (1 + |x|)
template <typename T>
struct SoftsignFunctor {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x / (static_cast<T>(1) + x.abs());
  }
};

// d(softsign(x))/dx = 1 / (1 + |x|)^2
// Taken from https://en.wikipedia.org/wiki/Activation_function
template <typename T>
struct SoftsignGradFunctor {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) =
        dy * (static_cast<T>(1) / (static_cast<T>(1) + x.abs()).square());
  }
};

template <typename Place, typename T, typename AttrType = T>
class BReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Output<framework::Tensor>("Y");
    auto t_min = static_cast<T>(context.Attr<AttrType>("t_min"));
    auto t_max = static_cast<T>(context.Attr<AttrType>("t_max"));
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();
    y.device(place) = x.cwiseMax(t_min).cwiseMin(t_max);
  }
};

template <typename Place, typename T, typename AttrType = T>
class BReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* dY = context.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto t_min = static_cast<T>(context.Attr<AttrType>("t_min"));
    auto t_max = static_cast<T>(context.Attr<AttrType>("t_max"));
    dX->mutable_data<T>(context.GetPlace());

    auto dy = framework::EigenVector<T>::Flatten(*dY);
    auto x = framework::EigenVector<T>::Flatten(*X);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();

    dx.device(place) = dy * ((x > t_min) * (x < t_max)).template cast<T>();
  }
};

template <typename Place, typename T, typename AttrType = T>
class SoftReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Output<framework::Tensor>("Y");
    auto threshold = static_cast<T>(context.Attr<AttrType>("threshold"));
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();
    auto temp = x.cwiseMax(-threshold).cwiseMin(threshold).eval();
    y.device(place) = (static_cast<T>(1) + temp.exp()).log();
  }
};

template <typename Place, typename T, typename AttrType = T>
class SoftReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Input<framework::Tensor>("Y");
    auto* dY = context.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto threshold = static_cast<T>(context.Attr<AttrType>("threshold"));
    dX->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto dy = framework::EigenVector<T>::Flatten(*dY);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();
    auto temp = ((x > -threshold) * (x < threshold)).template cast<T>().eval();
    dx.device(place) = dy * (static_cast<T>(1) - (-y).exp()) * temp;
  }
};

template <typename Place, typename T, typename AttrType = T>
class PowKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Output<framework::Tensor>("Y");
    auto factor = static_cast<T>(context.Attr<AttrType>("factor"));
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();
    y.device(place) = x.pow(factor);
  }
};

template <typename Place, typename T, typename AttrType = T>
class PowGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* dY = context.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto factor = static_cast<T>(context.Attr<AttrType>("factor"));
    dX->mutable_data<T>(context.GetPlace());

    auto dy = framework::EigenVector<T>::Flatten(*dY);
    auto x = framework::EigenVector<T>::Flatten(*X);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();

    dx.device(place) = dy * factor * x.pow(factor - static_cast<T>(1));
  }
};

template <typename Place, typename T, typename AttrType = T>
class STanhKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Output<framework::Tensor>("Y");
    auto scale_a = static_cast<T>(context.Attr<AttrType>("scale_a"));
    auto scale_b = static_cast<T>(context.Attr<AttrType>("scale_b"));
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();
    y.device(place) = scale_b * (scale_a * x).tanh();
  }
};

template <typename Place, typename T, typename AttrType = T>
class STanhGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* dY = context.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto scale_a = static_cast<T>(context.Attr<AttrType>("scale_a"));
    auto scale_b = static_cast<T>(context.Attr<AttrType>("scale_b"));
    dX->mutable_data<T>(context.GetPlace());

    auto dy = framework::EigenVector<T>::Flatten(*dY);
    auto x = framework::EigenVector<T>::Flatten(*X);
    auto dx = framework::EigenVector<T>::Flatten(*dX);
    auto place = context.GetEigenDevice<Place>();

    auto temp = (scale_a * x).tanh() * (scale_a * x).tanh();
    dx.device(place) = dy * scale_a * scale_b * (static_cast<T>(1) - temp);
  }
};

}  // namespace operators
}  // namespace paddle
