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

// Originally: logsigmoid(x) = -log (1 + exp(-x))
// For numerical stability, we can use the log-sum-exp trick:
// https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
// We can rewrite the above equation as:
// y = -log( exp(0) + exp(-x)) [since exp(0) = 1]
//   = -log( exp(max(-x, 0) - max(-x, 0)) + exp(-x + max(-x, 0) - max(-x, 0)))
//   = -log( exp(max(-x, 0)) * exp(-max(-x, 0)) - exp(max(-x, 0)) * exp(-x -
//           max(-x, 0)))
//   = -log( exp(max(-x, 0)) * (exp(-max(-x, 0)) + exp(-x - max(-x, 0))))
//   = -log( exp(max(-x, 0)) - log(exp(-max(-x, 0)) + exp(-x - max(-x, 0)))
//
// Hence, logsigmoid(x) = - (max(-x, 0) + log(exp(-max(-x, 0))
// + exp(-x - max(-x, 0))))
template <typename T>
struct LogSigmoidFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    auto temp = (-x).cwiseMax(static_cast<T>(0));  // temp = max(-x, 0)
    y.device(d) = -temp - (((-temp).exp() + (-x - temp).exp()).log());
  }
};

// Originally: f' = exp(-x) / (1 + exp(-x))
// For numerical stability: f' = exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) +
// exp(-x - max(-x, 0)))
template <typename T>
struct LogSigmoidGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto temp = (-x).cwiseMax(static_cast<T>(0));  // temp = max(-x, 0)
    dx.device(d) =
        dy * ((-x - temp).exp() / ((-temp).exp() + (-x - temp).exp()));
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

// tanhshrink(x) = x - tanh(x)
// where tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct HardShrinkFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    auto temp1 = (x < static_cast<T>(threshold * -1)).template cast<T>().eval();
    auto temp2 = (x > static_cast<T>(threshold)).template cast<T>().eval();
    y.device(d) = x * (temp1 + temp2);
  }
};

template <typename T>
struct HardShrinkGradFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto temp1 = (x < static_cast<T>(threshold * -1)).template cast<T>().eval();
    auto temp2 = (x > static_cast<T>(threshold)).template cast<T>().eval();
    dx.device(d) = dy * (temp1 + temp2).template cast<T>();
  }
};

// softshrink(x) = x - lambda, if x > lambda; x + lambda, if x < -lambda; 0
// otherwise
template <typename T>
struct SoftShrinkFunctor : public BaseActivationFunctor<T> {
  float lambda;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    auto lambdaT = static_cast<T>(lambda);
    auto temp1 = (x > lambdaT).template cast<T>().eval();
    auto temp2 = (x < -lambdaT).template cast<T>().eval();
    y.device(d) = temp1 * (x - lambdaT) + temp2 * (x + lambdaT);
  }
};

template <typename T>
struct SoftShrinkGradFunctor : public BaseActivationFunctor<T> {
  float lambda;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto lambdaT = static_cast<T>(lambda);
    auto temp1 = (x > lambdaT).template cast<T>().eval();
    auto temp2 = (x < -lambdaT).template cast<T>().eval();
    dx.device(d) = dy * (temp1 + temp2).template cast<T>();
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
    y.device(d) =
        x.cwiseMax(static_cast<T>(t_min)).cwiseMin(static_cast<T>(t_max));
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
    dx.device(d) = dy *
                   ((x > static_cast<T>(t_min)) * (x < static_cast<T>(t_max)))
                       .template cast<T>();
  }
};

// relu6(x) = min(max(0, x), 6)
template <typename T>
struct Relu6Functor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) =
        x.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(threshold));
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
    dx.device(d) = dy *
                   ((x > static_cast<T>(0)) * (x < static_cast<T>(threshold)))
                       .template cast<T>();
  }
};

// softplus(x) = log(1 + exp(x))
// When x is a very large positive number, exp(x) may explode to inf,
// Using trick below for numerical stability
// https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
// Then: softplus(x) = max(x, 0) + log(exp(-max(x, 0)) + exp(x - max(x, 0)))
template <typename T>
struct SoftplusFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) {
    auto temp = x.cwiseMax(static_cast<T>(0));  // temp = max(x, 0)
    y.device(d) = temp + (((-temp).exp() + (x - temp).exp()).log());
  }
};

// d(softplus(x))/dx = exp(x) / (1 + exp(x))
// For numerical stability:
// d(softplus(x))/dx = exp(x - max(x, 0)) / (exp(-max(x, 0)) +
// exp(x - max(x, 0)))
template <typename T>
struct SoftplusGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) {
    auto temp = x.cwiseMax(static_cast<T>(0));  // temp = max(x, 0)
    dx.device(d) = dy * ((x - temp).exp() / ((-temp).exp() + (x - temp).exp()));
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
    auto tmp = static_cast<T>(threshold);
    auto temp = x.cwiseMax(-tmp).cwiseMin(tmp);
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
    auto tmp = static_cast<T>(threshold);
    auto temp = ((x > -tmp) * (x < tmp)).template cast<T>().eval();
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
    y.device(d) = x.cwiseMax(static_cast<T>(alpha) * x);
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
    auto temp1 = static_cast<T>(alpha) *
                 (x < static_cast<T>(0)).template cast<T>().eval();
    auto temp2 = (x >= static_cast<T>(0)).template cast<T>().eval();
    dx.device(d) = dy * (temp1 + temp2).template cast<T>();
  }
};

template <typename T>
struct ELUFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.cwiseMax(static_cast<T>(0)) +
                  (static_cast<T>(alpha) * (x.exp() - static_cast<T>(1)))
                      .cwiseMin(static_cast<T>(0));
  }
};

template <typename T>
struct ELUGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) = dy * (x > static_cast<T>(0)).template cast<T>() +
                   dy * (y + static_cast<T>(alpha)) *
                       (x < static_cast<T>(0)).template cast<T>();
  }
};

// FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5198
template <typename T>
struct PowFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }
  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    y.device(d) = x.pow(static_cast<T>(factor));
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
    dx.device(d) = dy * static_cast<T>(factor) *
                   x.pow(static_cast<T>(factor - static_cast<T>(1)));
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
    y.device(d) =
        static_cast<T>(scale_b) * (static_cast<T>(scale_a) * x).tanh();
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
    auto a = static_cast<T>(scale_a);
    auto b = static_cast<T>(scale_b);
    auto temp = (a * x).tanh() * (a * x).tanh();
    dx.device(d) = dy * a * b * (static_cast<T>(1) - temp);
  }
};

template <typename T>
struct ThresholdedReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    auto th = static_cast<T>(threshold);
    y.device(d) = (x > th).template cast<T>() * x;
  }
};

template <typename T>
struct ThresholdedReluGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    auto th = static_cast<T>(threshold);
    dx.device(d) = dy * (x > th).template cast<T>();
  }
};

template <typename T>
struct HardSigmoidFunctor : public BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  template <typename Device, typename X, typename Y>
  void operator()(Device d, X x, Y y) const {
    auto temp = x * static_cast<T>(slope) + static_cast<T>(offset);
    y.device(d) = temp.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(1));
  }
};

template <typename T>
struct HardSigmoidGradFunctor : public BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  template <typename Device, typename X, typename Y, typename dY, typename dX>
  void operator()(Device d, X x, Y y, dY dy, dX dx) const {
    dx.device(d) =
        dy *
        ((y > static_cast<T>(0)) * (y < static_cast<T>(1))).template cast<T>() *
        static_cast<T>(slope);
  }
};

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_KERNEL_FUNCTOR(__macro)                             \
  __macro(sigmoid, SigmoidFunctor, SigmoidGradFunctor);              \
  __macro(logsigmoid, LogSigmoidFunctor, LogSigmoidGradFunctor);     \
  __macro(exp, ExpFunctor, ExpGradFunctor);                          \
  __macro(relu, ReluFunctor, ReluGradFunctor);                       \
  __macro(tanh, TanhFunctor, TanhGradFunctor);                       \
  __macro(softshrink, SoftShrinkFunctor, SoftShrinkGradFunctor);     \
  __macro(sqrt, SqrtFunctor, SqrtGradFunctor);                       \
  __macro(abs, AbsFunctor, AbsGradFunctor);                          \
  __macro(reciprocal, ReciprocalFunctor, ReciprocalGradFunctor);     \
  __macro(log, LogFunctor, LogGradFunctor);                          \
  __macro(square, SquareFunctor, SquareGradFunctor);                 \
  __macro(brelu, BReluFunctor, BReluGradFunctor);                    \
  __macro(soft_relu, SoftReluFunctor, SoftReluGradFunctor);          \
  __macro(pow, PowFunctor, PowGradFunctor);                          \
  __macro(stanh, STanhFunctor, STanhGradFunctor);                    \
  __macro(softplus, SoftplusFunctor, SoftplusGradFunctor);           \
  __macro(softsign, SoftsignFunctor, SoftsignGradFunctor);           \
  __macro(relu6, Relu6Functor, Relu6GradFunctor);                    \
  __macro(leaky_relu, LeakyReluFunctor, LeakyReluGradFunctor);       \
  __macro(tanh_shrink, TanhShrinkFunctor, TanhShrinkGradFunctor);    \
  __macro(elu, ELUFunctor, ELUGradFunctor);                          \
  __macro(hard_shrink, HardShrinkFunctor, HardShrinkGradFunctor);    \
  __macro(hard_sigmoid, HardSigmoidFunctor, HardSigmoidGradFunctor); \
  __macro(thresholded_relu, ThresholdedReluFunctor, ThresholdedReluGradFunctor);
