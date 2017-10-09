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

template <typename Place, typename Functor>
class ActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;

  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Output<framework::Tensor>("Y");
    Y->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(*X);
    auto y = framework::EigenVector<T>::Flatten(*Y);
    auto place = context.GetEigenDevice<Place>();
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(place, x, y);
  }
};

template <typename Place, typename Functor>
class ActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
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
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(place, x, y, dy, dx);
  }
};

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char*, float*>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

// sigmoid(x) = 1 / (1 + exp(-x))
template <typename T>
struct SigmoidFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
  }
};

template <typename T>
struct SigmoidGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * y * (static_cast<T>(1) - y);
  }
};

// exp(x) = e^x
template <typename T>
struct ExpFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.exp();
  }
};

template <typename T>
struct ExpGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * y;
  }
};

// relu(x) = max(x, 0)
template <typename T>
struct ReluFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.cwiseMax(static_cast<T>(0));
  }
};

template <typename T>
struct ReluGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * (x > static_cast<T>(0)).template cast<T>();
  }
};

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.tanh();
  }
};

template <typename T>
struct TanhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * (static_cast<T>(1) - y * y);
  }
};

// tanhshrink(x) = x - tanh(x)
// where tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhShrinkFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x - x.tanh();
  }
};

template <typename T>
struct TanhShrinkGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * (x.tanh() * x.tanh());
  }
};

// sqrt(x) = x^(1/2)
template <typename T>
struct SqrtFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.sqrt();
  }
};

template <typename T>
struct SqrtGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    const Y y_conj = Eigen::numext::conj(y);
    dx.device(d) = static_cast<T>(0.5) * dy / y_conj;
  }
};

// abs(x) = |x|
template <typename T>
struct AbsFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.abs();
  }
};

template <typename T>
struct AbsGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * x.sign();
  }
};

// reciprocal(x) = 1 / x
template <typename T>
struct ReciprocalFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = static_cast<T>(1) / x;
  }
};

template <typename T>
struct ReciprocalGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * static_cast<T>(-1) * y * y;
  }
};

// log(x) = natural logarithm of x
template <typename T>
struct LogFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.log();
  }
};

template <typename T>
struct LogGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * (static_cast<T>(1) / x);
  }
};

// square(x) = x^2
template <typename T>
struct SquareFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.square();
  }
};

template <typename T>
struct SquareGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * static_cast<T>(2) * x;
  }
};

template <typename T>
struct BReluFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;

  // NOTE: Explicit hides the `BaseActivationFunctor<T>::GetAttrs`
  // not polymorphism for speed.
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.cwiseMax(t_min).cwiseMin(t_max);
  }
};

template <typename T>
struct BReluGradFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * ((x > t_min) * (x < t_max)).template cast<T>();
  }
};

// relu6(x) = min(max(0, x), 6)
template <typename T>
struct Relu6Functor : public BaseActivationFunctor<T> {
  float threshold;

  // NOTE: Explicit hides the `BaseActivationFunctor<T>::GetAttrs`
  // not polymorphism for speed.
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.cwiseMax(static_cast<T>(0)).cwiseMin(threshold);
  }
};

template <typename T>
struct Relu6GradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) =
        dy * ((x > static_cast<T>(0)) * (x < threshold)).template cast<T>();
  }
};

// softsign(x) = x / (1 + |x|)
template <typename T>
struct SoftsignFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    y.device(d) = x / (static_cast<T>(1) + x.abs());
  }
};

// d(softsign(x))/dx = 1 / (1 + |x|)^2
// Taken from https://en.wikipedia.org/wiki/Activation_function
template <typename T>
struct SoftsignGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    dx.device(d) =
        dy * (static_cast<T>(1) / (static_cast<T>(1) + x.abs()).square());
  }
};

template <typename T>
struct SoftReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    auto temp = x.cwiseMax(-threshold).cwiseMin(threshold);
    y.device(d) = (static_cast<T>(1) + temp.exp()).log();
  }
};

template <typename T>
struct SoftReluGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto temp = ((x > -threshold) * (x < threshold)).template cast<T>().eval();
    dx.device(d) = dy * (static_cast<T>(1) - (-y).exp()) * temp;
  }
};

template <typename T>
struct LeakyReluFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.cwiseMax(alpha * x);
  }
};

template <typename T>
struct LeakyReluGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto temp1 = alpha * (x < static_cast<T>(0)).template cast<T>().eval();
    auto temp2 = (x >= static_cast<T>(0)).template cast<T>().eval();
    dx.device(d) = dy * (temp1 + temp2).template cast<T>();
  }
};

template <typename T>
struct PowFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.pow(factor);
  }
};

template <typename T>
struct PowGradFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * factor * x.pow(factor - static_cast<T>(1));
  }
};

template <typename T>
struct STanhFunctor : public BaseActivationFunctor<T> {
  float scale_a;
  float scale_b;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = scale_b * (scale_a * x).tanh();
  }
};

template <typename T>
struct STanhGradFunctor : public BaseActivationFunctor<T> {
  float scale_a;
  float scale_b;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto temp = (scale_a * x).tanh() * (scale_a * x).tanh();
    dx.device(d) = dy * scale_a * scale_b * (static_cast<T>(1) - temp);
  }
};

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_KERNEL_FUNCTOR(__macro)                         \
  __macro(sigmoid, SigmoidFunctor, SigmoidGradFunctor);          \
  __macro(exp, ExpFunctor, ExpGradFunctor);                      \
  __macro(relu, ReluFunctor, ReluGradFunctor);                   \
  __macro(tanh, TanhFunctor, TanhGradFunctor);                   \
  __macro(sqrt, SqrtFunctor, SqrtGradFunctor);                   \
  __macro(abs, AbsFunctor, AbsGradFunctor);                      \
  __macro(reciprocal, ReciprocalFunctor, ReciprocalGradFunctor); \
  __macro(log, LogFunctor, LogGradFunctor);                      \
  __macro(square, SquareFunctor, SquareGradFunctor);             \
  __macro(brelu, BReluFunctor, BReluGradFunctor);                \
  __macro(soft_relu, SoftReluFunctor, SoftReluGradFunctor);      \
  __macro(pow, PowFunctor, PowGradFunctor);                      \
  __macro(stanh, STanhFunctor, STanhGradFunctor);                \
  __macro(softsign, SoftsignFunctor, SoftsignGradFunctor);       \
  __macro(relu6, Relu6Functor, Relu6GradFunctor);                \
  __macro(leaky_relu, LeakyReluFunctor, LeakyReluGradFunctor);   \
  __macro(tanh_shrink, TanhShrinkFunctor, TanhShrinkGradFunctor)
