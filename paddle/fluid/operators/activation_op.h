/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cmath>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <type_traits>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace paddle {
namespace operators {

using framework::To32BitIndex;

using ActBwdOpFwdDeps = phi::funcs::ActBwdOpFwdDeps;

/* The following operator can be used to process SelectedRows, because the
 * output of those operator for zero is zero too.
 */
static std::unordered_set<std::string> CanBeUsedBySelectedRows = {
    "abs", "abs_grad", "square", "square_grad", "sqrt", "sqrt_grad"};

inline void ExtractActivationTensor(const framework::ExecutionContext& context,
                                    const framework::Tensor** X,
                                    framework::Tensor** Out) {
  auto x_var = context.InputVar("X");
  auto out_var = context.OutputVar("Out");
  PADDLE_ENFORCE_NOT_NULL(x_var,
                          platform::errors::NotFound(
                              "Cannot get input Variable X, variable name = %s",
                              context.InputName("X")));
  PADDLE_ENFORCE_NOT_NULL(
      out_var, platform::errors::NotFound(
                   "Cannot get output Variable Out, variable name = %s",
                   context.OutputName("Out")));
  if (CanBeUsedBySelectedRows.count(context.Type())) {
    *X = paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_var);
    *Out = paddle::framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(
        out_var);
  } else {
    *X = context.Input<framework::Tensor>("X");
    *Out = context.Output<framework::Tensor>("Out");
  }

  PADDLE_ENFORCE_NOT_NULL(*Out, platform::errors::NotFound(
                                    "Cannot get the tensor from the Variable "
                                    "Output(Out), variable name = %s",
                                    context.OutputName("Out")));
}

template <ActBwdOpFwdDeps kDepValue>
inline void ExtractActivationGradTensor(
    const framework::ExecutionContext& context, const framework::Tensor** X,
    const framework::Tensor** Out, const framework::Tensor** dOut,
    framework::Tensor** dX) {
  auto out_grad_var = context.InputVar(framework::GradVarName("Out"));
  auto x_grad_var = context.OutputVar(framework::GradVarName("X"));
  const framework::Variable* out_var = nullptr;

  if (static_cast<int>(kDepValue) &
      static_cast<int>(ActBwdOpFwdDeps::kDepOut)) {
    out_var = context.InputVar("Out");
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "Cannot get input Variable Out, variable name = %s",
                     context.InputName("Out")));
  }

  PADDLE_ENFORCE_NOT_NULL(
      out_grad_var, platform::errors::NotFound(
                        "Cannot get input Variable %s, variable name = %s",
                        framework::GradVarName("Out"),
                        context.InputName(framework::GradVarName("Out"))));
  PADDLE_ENFORCE_NOT_NULL(
      x_grad_var, platform::errors::NotFound(
                      "Cannot get output Variable %s, variable name = %s",
                      framework::GradVarName("X"),
                      context.OutputName(framework::GradVarName("X"))));

  if (CanBeUsedBySelectedRows.count(context.Type())) {
    *dOut = paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(
        *out_grad_var);
    *dX = paddle::framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(
        x_grad_var);

    if (out_var) {
      *Out =
          paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(*out_var);
    } else {
      *Out = *dOut;  // fake out
    }

  } else {
    *Out = context.Input<framework::Tensor>("Out");
    *dOut = context.Input<framework::Tensor>(framework::GradVarName("Out"));
    *dX = context.Output<framework::Tensor>(framework::GradVarName("X"));

    if (out_var) {
      *Out = &(out_var->Get<framework::LoDTensor>());
    } else {
      *Out = *dOut;  // fake out
    }
  }

  PADDLE_ENFORCE_NOT_NULL(*dX,
                          platform::errors::NotFound(
                              "Cannot get the tensor from the Variable "
                              "Output(Out), variable name = %s",
                              context.OutputName(framework::GradVarName("X"))));

  if (static_cast<int>(kDepValue) & static_cast<int>(ActBwdOpFwdDeps::kDepX)) {
    auto x_var = context.InputVar("X");
    PADDLE_ENFORCE_NOT_NULL(x_var, platform::errors::NotFound(
                                       "Cannot get the tensor from the "
                                       "Variable Input(X), variable name = %s",
                                       context.InputName("X")));
    if (CanBeUsedBySelectedRows.count(context.Type())) {
      *X = paddle::framework::GetLoDTensorOrSelectedRowsValueFromVar(*x_var);
    } else {
      *X = context.Input<framework::Tensor>("X");
    }
  } else {
    VLOG(10) << " Inplace activation of Op : " << context.Type();
    *X = *dX;
  }
}

template <typename DeviceContext, typename Functor>
class ActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;

  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* X = nullptr;
    framework::Tensor* Out = nullptr;
    ExtractActivationTensor(context, &X, &Out);
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "Activation"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "Activation"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    // use 32bit index to speed up computation
    bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
    bool is_gpu_place = platform::is_gpu_place(context.GetPlace());
    if (use_32bit_index && is_gpu_place) {
      functor(*place, To32BitIndex(x), To32BitIndex(out));
    } else {
      functor(*place, x, out);
    }
  }
};

template <typename DeviceContext, typename Functor>
class ActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor *X, *Out, *dOut;
    framework::Tensor* dX = nullptr;
    X = Out = dOut = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(context, &X, &Out, &dOut,
                                                    &dX);
    dX->mutable_data<T>(context.GetPlace());
    auto dout = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "ActivationGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "ActivationGrad"));
    auto dx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Input", "X@GRAD", "ActivationGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "ActivationGrad"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    // use 32bit index to speed up computation
    bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
    bool is_gpu_place = platform::is_gpu_place(context.GetPlace());
    if (use_32bit_index && is_gpu_place) {
      functor(*place, To32BitIndex(x), To32BitIndex(out), To32BitIndex(dout),
              To32BitIndex(dx));
    } else {
      functor(*place, x, out, dout, dx);
    }
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
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = static_cast<T>(1) / (static_cast<T>(1) + (-x).exp());
  }
};

#define USE_PHI_FUNCTOR(name)                         \
  template <typename T>                               \
  using name##Functor = phi::funcs::name##Functor<T>; \
  template <typename T>                               \
  using name##GradFunctor = phi::funcs::name##GradFunctor<T>;

#define USE_PHI_DOUBLE_GRAD_FUNCTOR(name) \
  template <typename T>                   \
  using name##GradGradFunctor = phi::funcs::name##GradGradFunctor<T>;

#define USE_PHI_TRIPLE_GRAD_FUNCTOR(name) \
  template <typename T>                   \
  using name##TripleGradFunctor = phi::funcs::name##TripleGradFunctor<T>;

USE_PHI_FUNCTOR(Cos)
USE_PHI_FUNCTOR(Tan)
USE_PHI_FUNCTOR(Acos)
USE_PHI_FUNCTOR(Sin)
USE_PHI_FUNCTOR(Asin)
USE_PHI_FUNCTOR(Atan)
USE_PHI_FUNCTOR(Sinh)
USE_PHI_FUNCTOR(Cosh)
USE_PHI_FUNCTOR(Asinh)
USE_PHI_FUNCTOR(Acosh)
USE_PHI_FUNCTOR(Atanh)
USE_PHI_FUNCTOR(Tanh)
USE_PHI_FUNCTOR(Exp)
USE_PHI_DOUBLE_GRAD_FUNCTOR(Tanh)
USE_PHI_TRIPLE_GRAD_FUNCTOR(Tanh)
USE_PHI_FUNCTOR(BRelu)
USE_PHI_FUNCTOR(ThresholdedRelu)
USE_PHI_FUNCTOR(LeakyRelu)
USE_PHI_DOUBLE_GRAD_FUNCTOR(LeakyRelu)
USE_PHI_FUNCTOR(HardShrink)
USE_PHI_FUNCTOR(SoftShrink)
USE_PHI_FUNCTOR(TanhShrink)
USE_PHI_FUNCTOR(Silu)
USE_PHI_FUNCTOR(ELU)
USE_PHI_DOUBLE_GRAD_FUNCTOR(ELU)

template <typename T>
using ELUGradNegativeAlphaFunctor = phi::funcs::ELUGradNegativeAlphaFunctor<T>;

template <typename T>
struct SigmoidGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out * (static_cast<T>(1) - out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

/*
    Out
    DOut -> SigmoidGradGrad -> DOutNew
    DDX                        DDOut

    DDOut = (1-Out)*Out*DDX
    DOutNew = (1-2*Out)*DOut*DDX
*/
template <typename T>
struct SigmoidGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* Out,
                  const framework::Tensor* ddX, const framework::Tensor* dOut,
                  framework::Tensor* dOutNew, framework::Tensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SigmoidGradGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "SigmoidGradGrad"));

    if (dOutNew) {
      auto dout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Input", "DOut", "SigmoidGradGrad"));
      auto dout_new = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOutNew, "Output", "DOutNew", "SigmoidGradGrad"));
      dout_new.device(*d) =
          (static_cast<T>(1) - static_cast<T>(2) * out) * dout * ddx;
    }
    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SigmoidGradGrad"));
      ddout.device(*d) = (static_cast<T>(1) - out) * out * ddx;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

/*
    Out
    DOut                            D_Dout
    DDx     -> SigmoidTripleGrad -> D_DDx
    D_DDout                         d_OutNew
    D_Dout_new

    D_Dout = (1-2*Out)*DDx*D_Dout_new
    D_DDx = (1-Out)*Out*D_DDout + (1-2*Out)*DOut*D_Dout_new
    D_OutNew = (DDx-2*Out*DDx)*D_DDout - 2*DOut*DDx*D_Dout_new

    Out, DDX, DOut, D_DDOut, D_DOut_New   // input
    D_OutNew, D_DOut, D_DDx               // output
*/
template <typename T>
struct SigmoidTripleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* Out,
                  const framework::Tensor* ddX, const framework::Tensor* dOut,
                  const framework::Tensor* d_DDOut,
                  const framework::Tensor* d_dOut_New,
                  framework::Tensor* d_d_Out, framework::Tensor* d_Out_New,
                  framework::Tensor* d_DDx) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SigmoidTripleGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "SigmoidTripleGrad"));
    auto dout = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "DOut", "SigmoidTripleGrad"));
    auto d_ddOut = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "SigmoidTripleGrad"));
    auto d_dOutNew = framework::EigenVector<T>::Flatten(GET_DATA_SAFELY(
        d_dOut_New, "Input", "D_DOut_New", "SigmoidTripleGrad"));

    if (d_Out_New) {
      auto d_OutNew = framework::EigenVector<T>::Flatten(GET_DATA_SAFELY(
          d_Out_New, "Output", "D_OutNew", "SigmoidTripleGrad"));
      d_OutNew.device(*d) = (ddx - static_cast<T>(2) * out * ddx) * d_ddOut -
                            static_cast<T>(2) * dout * ddx * d_dOutNew;
    }
    if (d_d_Out) {
      auto d_dOut = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_d_Out, "Output", "D_DOut", "SigmoidTripleGrad"));
      d_dOut.device(*d) =
          (static_cast<T>(1) - static_cast<T>(2) * out) * ddx * d_dOutNew;
    }
    if (d_DDx) {
      auto d_ddx = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_DDx, "Output", "D_DDx", "SigmoidTripleGrad"));
      d_ddx.device(*d) =
          (static_cast<T>(1) - out) * out * d_ddOut +
          (static_cast<T>(1) - static_cast<T>(2) * out) * dout * d_dOutNew;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// Originally: logsigmoid(x) = -log (1 + exp(-x))
// For numerical stability, we can use the log-sum-exp trick:
// https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
// We can rewrite the above equation as:
// out = -log( exp(0) + exp(-x)) [since exp(0) = 1]
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
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp = (-x).cwiseMax(static_cast<T>(0));  // temp = max(-x, 0)
    out.device(d) = -temp - (((-temp).exp() + (-x - temp).exp()).log());
  }
};

// Originally: f' = exp(-x) / (1 + exp(-x))
// For numerical stability: f' = exp(-x - max(-x, 0)) / (exp(-max(-x, 0)) +
// exp(-x - max(-x, 0)))
template <typename T>
struct LogSigmoidGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto temp = (-x).cwiseMax(static_cast<T>(0));  // temp = max(-x, 0)
    dx.device(d) =
        dout * ((-x - temp).exp() / ((-temp).exp() + (-x - temp).exp()));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct Expm1GradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * out + dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// relu(x) = max(x, 0)

template <typename T>
using ReluCPUFunctor = phi::funcs::ReluCPUFunctor<T>;
template <typename T>
using ReluGradFunctor = phi::funcs::ReluGradFunctor<T>;

template <typename T>
using ReluGradGradFunctor = phi::funcs::ReluGradGradFunctor<T>;

template <typename T>
using ReluCUDAFunctor = phi::funcs::ReluCUDAFunctor<T>;

// tanhshrink(x) = x - tanh(x)
// where tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhShrinkFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x - x.tanh();
  }
};

template <typename T>
struct TanhShrinkGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (x.tanh() * x.tanh());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// tanhshrink(x) = x - tanh(x)
// where tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct HardShrinkFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp1 = x < static_cast<T>(threshold * -1.f);
    auto temp2 = x > static_cast<T>(threshold);
    out.device(d) = x * (temp1 || temp2).template cast<T>();
  }
};

template <typename T>
struct HardShrinkGradFunctor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto temp1 = x < static_cast<T>(threshold * -1.f);
    auto temp2 = x > static_cast<T>(threshold);
    dx.device(d) = dout * (temp1 || temp2).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// softshrink(x) = x - lambda, if x > lambda; x + lambda, if x < -lambda; 0
// otherwise
template <typename T>
struct SoftShrinkFunctor : public BaseActivationFunctor<T> {
  float lambda;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto lambdaT = static_cast<T>(lambda);
    auto temp1 = (x > lambdaT).template cast<T>();
    auto temp2 = (x < -lambdaT).template cast<T>();
    out.device(d) = temp1 * (x - lambdaT) + temp2 * (x + lambdaT);
  }
};

template <typename T>
struct SoftShrinkGradFunctor : public BaseActivationFunctor<T> {
  float lambda;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"lambda", &lambda}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto lambdaT = static_cast<T>(lambda);
    auto temp1 = (x > lambdaT).template cast<T>();
    auto temp2 = (x < -lambdaT).template cast<T>();
    dx.device(d) = dout * (temp1 + temp2).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SqrtGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = static_cast<T>(0.5) * dout / out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct RsqrtGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = static_cast<T>(-0.5) * dout * out * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// ceil(x) = ceiling(x)
template <typename T>
struct CeilFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.ceil();
  }
};

template <typename T>
struct ZeroGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = static_cast<T>(0) * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kNoDeps;
  }
};

// floor(x) = flooring(x)
template <typename T>
struct FloorFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.floor();
  }
};

// round(x) = [x]
template <typename T>
struct RoundFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.round();
  }
};

template <typename T>
struct ReciprocalGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(-1) * out * out;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// log(x) = natural logarithm of x
template <typename T>
struct LogFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.log();
  }
};

template <typename T>
struct LogGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<T>(1) / x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// log2(x) = logarithm to the base 2 of the elements of x
template <typename T>
struct Log2Functor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.log() / static_cast<T>(log(2));
  }
};

// the gradient of log2(x) is 1/(x*ln(2))
template <typename T>
struct Log2GradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (x * static_cast<T>(log(2)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// log10(x) = logarithm to the base 10 of the elements of x
template <typename T>
struct Log10Functor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.log() / static_cast<T>(log(10));
  }
};

// the gradient of log10(x) is 1/(x*ln(10))
template <typename T>
struct Log10GradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (x * static_cast<T>(log(10)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// log1p(x) = natural logarithm of x+1
template <typename T>
struct Log1pFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = (static_cast<T>(1) + x).log();
  }
};

template <typename T>
struct Log1pGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<T>(1) / (x + static_cast<T>(1)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SquareGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(2) * x;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// relu6(x) = min(max(0, x), 6)
template <typename T>
struct Relu6Functor : public BaseActivationFunctor<T> {
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
        x.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(threshold));
  }
};

template <typename T>
struct Relu6GradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        dout *
        ((out > static_cast<T>(0)) * (out < static_cast<T>(threshold)))
            .template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// HardSwish = min(max(0, x+3), 6) * x / 6
template <typename T>
struct HardSwishFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = (x + static_cast<T>(offset))
                        .cwiseMax(static_cast<T>(0))
                        .cwiseMin(static_cast<T>(threshold)) *
                    x / static_cast<T>(scale);
  }
};

template <typename T>
struct HardSwishGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  float scale;
  float offset;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}, {"scale", &scale}, {"offset", &offset}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto tmp = ((x + static_cast<T>(offset)) < static_cast<T>(threshold))
                   .template cast<T>();
    dx.device(d) =
        dout *
        (((x + static_cast<T>(offset)) > static_cast<T>(0)).template cast<T>() *
             (static_cast<T>(2) * x + static_cast<T>(offset)) /
             static_cast<T>(scale) * tmp +
         static_cast<T>(1) * (static_cast<T>(1) - tmp));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// For numerical stability, using the following formula instead of
// d(softplus(x))/dx = 1 / (1 + exp(-x))
// d(softplus(x))/dx = 1 / (1 + exp(-beta * x)) when beta * x <= threshold(beta
// = 1, threshold = 20 by default), otherwise x
template <typename T>
struct SoftplusGradFunctor : public BaseActivationFunctor<T> {
  float beta;
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}, {"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) {
    auto x_beta = static_cast<T>(beta) * x;
    dx.device(d) =
        (x_beta > static_cast<T>(threshold))
            .select(dout, dout / (static_cast<T>(1) + (-x_beta).exp()));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// dx = dout * (tanh(sp) + x * (1 - tanh(sp) ** 2) * (1 - exp(-sp)))
// sp = softplus(x)
template <typename T>
struct MishGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) {
    auto sp = (x > static_cast<T>(threshold))
                  .select(x, (static_cast<T>(1) + x.exp()).log());
    auto gsp = static_cast<T>(1) - (-sp).exp();
    auto tsp = sp.tanh();
    dx.device(d) = dout * (tsp + x * (static_cast<T>(1) - tsp * tsp) * gsp);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// d(softsign(x))/dx = 1 / (1 + |x|)^2
// Taken from https://en.wikipedia.org/wiki/Activation_function
template <typename T>
struct SoftsignGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) {
    dx.device(d) =
        dout * (static_cast<T>(1) / (static_cast<T>(1) + x.abs()).square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SoftReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto tmp = static_cast<T>(threshold);
    auto temp = x.cwiseMax(-tmp).cwiseMin(tmp);
    out.device(d) = (static_cast<T>(1) + temp.exp()).log();
  }
};

template <typename T>
struct SoftReluGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto tmp = static_cast<T>(threshold);
    auto temp = ((out > -tmp) * (out < tmp)).template cast<T>();
    dx.device(d) = dout * (static_cast<T>(1) - (-out).exp()) * temp;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename DeviceContext, typename T>
class ELUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<framework::Tensor>("X");
    auto* Out = context.Input<framework::Tensor>("Out");
    auto* dOut =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<framework::Tensor>(framework::GradVarName("X"));
    const float alpha = context.Attr<float>("alpha");
    dX->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "elu_grad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "elu_grad"));
    auto dout = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "dOut", "elu_grad"));
    auto dx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Output", "dX", "elu_grad"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();

    if (alpha > 0) {
      ELUGradFunctor<T> functor;
      functor.alpha = alpha;
      functor(*place, x, out, dout, dx);
    } else {
      ELUGradNegativeAlphaFunctor<T> functor;
      functor.alpha = alpha;
      functor(*place, x, out, dout, dx);
    }
  }
};

template <typename T>
struct CELUFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
        (x < static_cast<T>(0))
            .select(static_cast<T>(alpha) *
                        ((x / static_cast<T>(alpha)).exp() - static_cast<T>(1)),
                    x);
  }
};

template <typename T>
struct CELUGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto temp_a_pos = static_cast<T>(alpha > 0);
    auto temp_a_neg = static_cast<T>(alpha <= 0);
    auto temp_x_pos = (x > static_cast<T>(0)).template cast<T>();
    auto temp_x_neg = (x <= static_cast<T>(0)).template cast<T>();

    // dx = dout, if alpha > 0 and x > 0
    // dx = dout * (x/alpha).exp(), if alpha > 0 and x <= 0
    // dx = dout , if alpha < 0 and x > 0
    // dx = dout * (x/alpha).exp(), if alpha < 0 and x <=0
    dx.device(d) =
        dout * temp_a_pos * temp_x_pos +
        dout * (x / static_cast<T>(alpha)).exp() * temp_a_pos * temp_x_neg +
        dout * temp_a_neg * temp_x_pos +
        dout * (x / static_cast<T>(alpha)).exp() * temp_a_neg * temp_x_neg;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// FIXME(qijun) https://github.com/PaddlePaddle/Paddle/issues/5198
template <typename T>
struct PowFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.pow(static_cast<T>(factor));
  }
};

template <typename T>
struct PowGradFunctor : public BaseActivationFunctor<T> {
  float factor;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"factor", &factor}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(factor) *
                   x.pow(static_cast<T>(factor) - static_cast<T>(1));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LogitGradFunctor {
  template <typename Device, typename X, typename dOut, typename dX, typename P>
  void operator()(Device d, X x, dOut dout, dX dx, P p, float eps) const {
    // logit(x)' = 1/(x*(1-x))
    dx.device(d) =
        (x < static_cast<T>(eps) || x > static_cast<T>(1.0 - eps))
            .select(p.constant(static_cast<T>(0)),
                    dout * (static_cast<T>(1) / ((static_cast<T>(1) - x) * x)));
  }
};

template <typename T>
struct STanhGradFunctor : public BaseActivationFunctor<T> {
  float scale_a;
  float scale_b;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"scale_a", &scale_a}, {"scale_b", &scale_b}};
  }

  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto a = static_cast<T>(scale_a);
    auto b = static_cast<T>(scale_b);
    auto temp = (a * x).tanh() * (a * x).tanh();
    dx.device(d) = dout * a * b * (static_cast<T>(1) - temp);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct HardSigmoidFunctor : public BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto temp = x * static_cast<T>(slope) + static_cast<T>(offset);
    out.device(d) =
        temp.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(1));
  }
};

template <typename T>
struct HardSigmoidGradFunctor : public BaseActivationFunctor<T> {
  float slope;
  float offset;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"slope", &slope}, {"offset", &offset}};
  }
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout *
                   ((out > static_cast<T>(0)) * (out < static_cast<T>(1)))
                       .template cast<T>() *
                   static_cast<T>(slope);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct SwishFunctor : public BaseActivationFunctor<T> {
  float beta;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x / (static_cast<T>(1) + (static_cast<T>(-beta) * x).exp());
  }
};

template <typename T>
struct SwishGradFunctor : public BaseActivationFunctor<T> {
  float beta;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out fake_out, dOut dout, dX dx) const {
    auto temp1 = static_cast<T>(1) /
                 (static_cast<T>(1) + (static_cast<T>(-beta) * x).exp());
    auto out = x * temp1;
    auto temp2 = temp1 * (static_cast<T>(1) - (static_cast<T>(beta) * out));
    dx.device(d) = dout * ((static_cast<T>(beta) * out) + temp2);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct AbsGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* X,
                  const framework::Tensor* Out, const framework::Tensor* ddX,
                  framework::Tensor* ddOut, framework::Tensor* dOut,
                  framework::Tensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "AbsGradGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "AbsGradGrad"));
    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "AbsGradGrad"));
      ddout.device(*d) = ddx * x.sign();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CELUGradGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* X,
                  const framework::Tensor* ddX, framework::Tensor* ddOut,
                  const framework::Tensor* dOut, framework::Tensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "CELUGradGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "CELUGradGrad"));

    if (dX) {
      auto dx = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "CELUGradGrad"));
      auto dout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "CELUGradGrad"));
      dx.device(*d) = ddx * dout / static_cast<T>(alpha) *
                      (x / static_cast<T>(alpha)).exp() *
                      (x <= static_cast<T>(0)).template cast<T>();
    }

    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "CELUGradGrad"));
      ddout.device(*d) = ddx *
                         ((x > static_cast<T>(0)).template cast<T>() +
                          (x / static_cast<T>(alpha)).exp() *
                              (x <= static_cast<T>(0)).template cast<T>())
                             .template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct SqrtGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* Out,
                  const framework::Tensor* ddX, framework::Tensor* ddOut,
                  framework::Tensor* dOut, const framework::Tensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SqrtGradGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "SqrtGradGrad"));
    // sqrt GradGrad: ddy = 0.5 * ddx / y, dy = -1 * dx * ddx
    // calculate dy first, so ddy can inplace ddx
    if (dOut) {
      auto dx = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "SqrtGradGrad"));
      auto dout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "SqrtGradGrad"));
      dout.device(*d) = dx * ddx * static_cast<T>(-1) / out;
    }
    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SqrtGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(0.5) / out;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct RsqrtGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* Out,
                  const framework::Tensor* ddX, framework::Tensor* ddOut,
                  framework::Tensor* dOut, const framework::Tensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "RsqrtGradGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "RsqrtGradGrad"));

    // rsqrt GradGrad: ddy = -0.5 * ddx * y * y * y, dy = (3/y) * dx * ddx
    if (dOut) {
      auto dx = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "RsqrtGradGrad"));
      auto dout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "RsqrtGradGrad"));
      dout.device(*d) = (static_cast<T>(3.0) / out) * dx * ddx;
    }
    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "RsqrtGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(-0.5) * out * out * out;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct SquareGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* X,
                  const framework::Tensor* ddX, framework::Tensor* ddOut,
                  const framework::Tensor* dOut, framework::Tensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "SquareGradGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "SquareGradGrad"));
    // square GradGrad: ddy=2x*ddx, dx=2dy*ddx
    // calculate dx first, so ddy can inplace ddx
    if (dX) {
      auto dx = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "SquareGradGrad"));
      auto dout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "SquareGradGrad"));
      dx.device(*d) = ddx * static_cast<T>(2) * dout;
    }
    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "SquareGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(2) * x;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

// TODO(dengkaipeng): double gradient calculation for Square/Sqrt need
// DOut(dy) as input(not output), tensor extraction is different from
// others. Impliment extraction kernel seperately here.
inline void ExtractDoubleGradTensorWithInputDOut(
    const framework::ExecutionContext& ctx, const framework::Tensor** X,
    const framework::Tensor** ddX, framework::Tensor** dX,
    const framework::Tensor** dOut, framework::Tensor** ddOut) {
  // extract ddX(output), ddOut(input)
  auto ddx_var = ctx.InputVar("DDX");
  auto ddo_var = ctx.OutputVar("DDOut");
  PADDLE_ENFORCE_NOT_NULL(
      ddx_var, platform::errors::NotFound(
                   "Cannot get input Variable Out, variable name = %s",
                   ctx.InputName("DDX")));
  *ddX = ctx.Input<framework::Tensor>("DDX");
  if (ddo_var) {
    *ddOut = ctx.Output<framework::Tensor>("DDOut");
  }
  PADDLE_ENFORCE_NOT_NULL(
      ddX,
      platform::errors::NotFound(
          "Cannot get the tensor from the Variable DDX, variable name = %s",
          ctx.OutputName("DDX")));

  // extract x(input), dx(output)
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NOT_NULL(
      x_var, platform::errors::NotFound(
                 "Cannot get input Variable Out, variable name = %s",
                 ctx.InputName("X")));
  auto dx_var = ctx.OutputVar("DX");
  *X = ctx.Input<framework::Tensor>("X");
  if (dx_var) {
    *dX = ctx.Output<framework::Tensor>("DX");
  }

  // extract dOut(input)
  auto dout_var = ctx.InputVar("DOut");
  if (dout_var) {
    *dOut = ctx.Input<framework::Tensor>("DOut");
  }
}

template <typename DeviceContext, typename Functor>
class SigmoidDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *Out, *ddX, *dOut;
    framework::Tensor *dOutNew, *ddOut;
    Out = ddX = dOut = nullptr;
    dOutNew = ddOut = nullptr;
    // extract ddx(input) and out(input)
    ddX = ctx.Input<framework::Tensor>("DDX");
    Out = ctx.Input<framework::Tensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(
        ddX, platform::errors::NotFound(
                 "Cannot get input Variable ddX, variable name = %s",
                 ctx.InputName("DDX")));
    PADDLE_ENFORCE_NOT_NULL(
        Out, platform::errors::NotFound(
                 "Cannot get input Variable Out, variable name = %s",
                 ctx.InputName("Out")));
    // set output ddout
    ddOut = ctx.Output<framework::Tensor>("DDOut");
    // extract dOut(intput)
    dOut = ctx.Input<framework::Tensor>("DOut");
    PADDLE_ENFORCE_NOT_NULL(
        dOut, platform::errors::NotFound(
                  "Cannot get input Variable dOut, variable name = %s",
                  ctx.InputName("DOut")));
    dOutNew = ctx.Output<framework::Tensor>("DOutNew");
    if (dOutNew) dOutNew->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    auto& place = ctx.template device_context<DeviceContext>();
    Functor functor;
    functor(place, Out, ddX, dOut, dOutNew, ddOut);
  }
};

// Out, DDX, DOut, D_DDOut, D_DOut_New   // input
// D_OutNew, D_DOut, D_DDx               // output
template <typename DeviceContext, typename Functor>
class SigmoidTripleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *Out, *ddX, *dOut, *d_ddOut, *d_dOutNew;
    framework::Tensor *d_OutNew, *d_dOut, *d_ddx;
    Out = ddX = dOut = d_ddOut = d_dOutNew = nullptr;
    d_OutNew = d_dOut = d_ddx = nullptr;

    // extract ddx(input), out(input), dOut(input), d_ddOut(input),
    // d_dOutNew(input)
    ddX = ctx.Input<framework::Tensor>("DDX");
    Out = ctx.Input<framework::Tensor>("Out");
    dOut = ctx.Input<framework::Tensor>("DOut");
    d_ddOut = ctx.Input<framework::Tensor>("D_DDOut");
    d_dOutNew = ctx.Input<framework::Tensor>("D_DOut_New");

    PADDLE_ENFORCE_NOT_NULL(
        ddX, platform::errors::NotFound(
                 "Cannot get input Variable ddX, variable name = %s",
                 ctx.InputName("DDX")));
    PADDLE_ENFORCE_NOT_NULL(
        Out, platform::errors::NotFound(
                 "Cannot get input Variable Out, variable name = %s",
                 ctx.InputName("Out")));
    PADDLE_ENFORCE_NOT_NULL(
        dOut, platform::errors::NotFound(
                  "Cannot get input Variable dOut, variable name = %s",
                  ctx.InputName("DOut")));
    PADDLE_ENFORCE_NOT_NULL(
        d_ddOut, platform::errors::NotFound(
                     "Cannot get input Variable d_ddOut, variable name = %s",
                     ctx.InputName("D_DDOut")));
    PADDLE_ENFORCE_NOT_NULL(
        d_dOutNew,
        platform::errors::NotFound(
            "Cannot get input Variable d_dOutNew, variable name = %s",
            ctx.InputName("D_DOutNew")));

    // set output d_OutNew、d_dOut、d_ddx
    d_dOut = ctx.Output<framework::Tensor>("D_DOut");
    d_OutNew = ctx.Output<framework::Tensor>("D_OutNew");
    d_ddx = ctx.Output<framework::Tensor>("D_DDx");

    if (d_dOut) d_dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (d_OutNew) d_OutNew->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (d_ddx) d_ddx->mutable_data<T>(ddX->dims(), ctx.GetPlace());
    auto& place = ctx.template device_context<DeviceContext>();
    Functor functor;
    functor(place, Out, ddX, dOut, d_ddOut, d_dOutNew,  // input
            d_dOut, d_OutNew, d_ddx);                   // output
  }
};

template <typename DeviceContext, typename Functor>
class TanhDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *Out, *ddX, *dOut;
    framework::Tensor *dOutNew, *ddOut;
    Out = ddX = dOut = nullptr;
    dOutNew = ddOut = nullptr;

    // extract ddx(input) and out(input)
    auto ddx_var = ctx.InputVar("DDX");
    auto out_var = ctx.InputVar("Out");
    PADDLE_ENFORCE_NOT_NULL(
        ddx_var, platform::errors::NotFound(
                     "Cannot get input Variable ddx, variable name = %s",
                     ctx.InputName("DDX")));
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "Cannot get input Variable out, variable name = %s",
                     ctx.InputName("Out")));
    ddX = ctx.Input<framework::Tensor>("DDX");
    Out = ctx.Input<framework::Tensor>("Out");

    // set output ddout
    auto ddout_var = ctx.OutputVar("DDOut");
    if (ddout_var) {
      ddOut = ctx.Output<framework::Tensor>("DDOut");
    }

    // extract dOut(intput)
    auto dout_var = ctx.InputVar("DOut");
    PADDLE_ENFORCE_NOT_NULL(
        dout_var, platform::errors::NotFound(
                      "Cannot get input Variable dout_var, variable name = %s",
                      ctx.InputName("DOut")));
    dOut = ctx.Input<framework::Tensor>("DOut");

    // set output dout_new
    auto dout_new_var = ctx.OutputVar("DOutNew");
    if (dout_new_var) {
      dOutNew = ctx.Output<framework::Tensor>("DOutNew");
    }

    if (dOutNew) dOutNew->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    auto& place = ctx.template device_context<DeviceContext>();
    Functor functor;
    functor(place, Out, ddX, dOut, dOutNew, ddOut);
  }
};

template <typename DeviceContext, typename Functor>
class TanhTripeGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *Out, *ddX, *dOut, *d_ddOut, *d_dOutNew;
    framework::Tensor *d_OutNew, *d_dOut, *d_ddx;
    Out = ddX = dOut = d_ddOut = d_dOutNew = nullptr;
    d_OutNew = d_dOut = d_ddx = nullptr;

    // extract ddx(input), out(input), dOut(input), d_ddOut(input),
    // d_dOutNew(input)
    ddX = ctx.Input<framework::Tensor>("DDX");
    Out = ctx.Input<framework::Tensor>("Out");
    dOut = ctx.Input<framework::Tensor>("DOut");
    d_ddOut = ctx.Input<framework::Tensor>("D_DDOut");
    d_dOutNew = ctx.Input<framework::Tensor>("D_DOut_New");

    PADDLE_ENFORCE_NOT_NULL(
        ddX, platform::errors::NotFound(
                 "Cannot get input Variable ddX, variable name = %s",
                 ctx.InputName("DDX")));
    PADDLE_ENFORCE_NOT_NULL(
        Out, platform::errors::NotFound(
                 "Cannot get input Variable Out, variable name = %s",
                 ctx.InputName("Out")));
    PADDLE_ENFORCE_NOT_NULL(
        dOut, platform::errors::NotFound(
                  "Cannot get input Variable dOut, variable name = %s",
                  ctx.InputName("DOut")));
    PADDLE_ENFORCE_NOT_NULL(
        d_ddOut, platform::errors::NotFound(
                     "Cannot get input Variable d_ddOut, variable name = %s",
                     ctx.InputName("D_DDOut")));
    PADDLE_ENFORCE_NOT_NULL(
        d_dOutNew,
        platform::errors::NotFound(
            "Cannot get input Variable d_dOutNew, variable name = %s",
            ctx.InputName("D_DOutNew")));

    // set output d_OutNew、d_dOut、d_ddx
    d_dOut = ctx.Output<framework::Tensor>("D_DOut");
    d_OutNew = ctx.Output<framework::Tensor>("D_OutNew");
    d_ddx = ctx.Output<framework::Tensor>("D_DDx");

    if (d_dOut) d_dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (d_OutNew) d_OutNew->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (d_ddx) d_ddx->mutable_data<T>(ddX->dims(), ctx.GetPlace());
    auto& place = ctx.template device_context<DeviceContext>();
    Functor functor;
    functor(place, Out, ddX, dOut, d_ddOut, d_dOutNew,  // input
            d_dOut, d_OutNew, d_ddx);                   // output
  }
};

template <typename DeviceContext, typename Functor>
class SquareDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *X, *ddX, *dOut;
    X = ddX = dOut = nullptr;
    framework::Tensor *dX, *ddOut;
    dX = ddOut = nullptr;

    ExtractDoubleGradTensorWithInputDOut(ctx, &X, &ddX, &dX, &dOut, &ddOut);

    if (dX) dX->mutable_data<T>(X->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(ctx.GetPlace());

    auto& place = ctx.template device_context<DeviceContext>();

    Functor functor;
    functor(place, X, ddX, ddOut, dOut, dX);
  }
};

template <typename DeviceContext, typename Functor>
class LogDoubleGradKernel
    : public SquareDoubleGradKernel<DeviceContext, Functor> {};

template <typename DeviceContext, typename Functor>
class ELUDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *X, *ddX, *dOut;
    X = ddX = dOut = nullptr;
    framework::Tensor *dX, *ddOut;
    dX = ddOut = nullptr;

    ExtractDoubleGradTensorWithInputDOut(ctx, &X, &ddX, &dX, &dOut, &ddOut);

    if (dX) dX->mutable_data<T>(X->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(ctx.GetPlace());

    auto& place = ctx.template device_context<DeviceContext>();

    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
    functor(place, X, ddX, ddOut, dOut, dX);
  }
};

template <typename DeviceContext, typename Functor>
class CELUDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *X, *ddX, *dOut;
    X = ddX = dOut = nullptr;
    framework::Tensor *dX, *ddOut;
    dX = ddOut = nullptr;

    ExtractDoubleGradTensorWithInputDOut(ctx, &X, &ddX, &dX, &dOut, &ddOut);

    if (dX) dX->mutable_data<T>(X->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(ctx.GetPlace());

    auto& place = ctx.template device_context<DeviceContext>();

    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
    functor(place, X, ddX, ddOut, dOut, dX);
  }
};

template <typename DeviceContext, typename Functor>
class SqrtDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *Out, *dX, *ddX;
    Out = dX = ddX = nullptr;
    framework::Tensor *ddOut, *dOut;
    ddOut = dOut = nullptr;

    // extract ddx(input), ddout(output)
    auto ddx_var = ctx.InputVar("DDX");
    auto ddo_var = ctx.OutputVar("DDOut");
    PADDLE_ENFORCE_NOT_NULL(
        ddx_var, platform::errors::NotFound(
                     "Cannot get input Variable DDX, variable name = %s",
                     ctx.InputName("DDX")));
    ddX = ctx.Input<framework::Tensor>("DDX");
    if (ddo_var) {
      ddOut = ctx.Output<framework::Tensor>("DDOut");
    }
    PADDLE_ENFORCE_NOT_NULL(
        ddX, platform::errors::NotFound(
                 "Cannot get input Variable DDX, variable name = %s",
                 ctx.InputName("DDX")));

    // extract out(input), dout(output)
    auto out_var = ctx.InputVar("Out");
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "Cannot get input Variable Out, variable name = %s",
                     ctx.InputName("Out")));
    auto dout_var = ctx.OutputVar("DOut");
    Out = ctx.Input<framework::Tensor>("Out");
    if (dout_var) {
      dOut = ctx.Output<framework::Tensor>("DOut");
    }

    // extract dx(input)
    auto dx_var = ctx.InputVar("DX");
    PADDLE_ENFORCE_NOT_NULL(
        dx_var, platform::errors::NotFound(
                    "Cannot get input Variable DX, variable name = %s",
                    ctx.InputName("DX")));
    if (dx_var) {
      dX = ctx.Input<framework::Tensor>("DX");
    }

    if (dOut) dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());

    auto& place = ctx.template device_context<DeviceContext>();

    Functor functor;
    functor(place, Out, ddX, ddOut, dOut, dX);
  }
};

// rsqrt Grad: dx = -0.5 * dy * y * y * y
// rsqrt GradGrad: ddy = -0.5 * ddx * y * y * y, dy = (3 / y) * dx * ddx
template <typename DeviceContext, typename Functor>
class RsqrtDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *Out, *dX, *ddX;
    Out = dX = ddX = nullptr;
    framework::Tensor *ddOut, *dOut;
    ddOut = dOut = nullptr;

    // extract ddx(input), ddout(output)
    auto ddx_var = ctx.InputVar("DDX");
    auto ddo_var = ctx.OutputVar("DDOut");
    PADDLE_ENFORCE_NOT_NULL(
        ddx_var, platform::errors::NotFound(
                     "Cannot get input Variable DDX, variable name = %s",
                     ctx.InputName("DDX")));
    ddX = ctx.Input<framework::Tensor>("DDX");
    if (ddo_var) {
      ddOut = ctx.Output<framework::Tensor>("DDOut");
    }
    PADDLE_ENFORCE_NOT_NULL(
        ddX, platform::errors::NotFound(
                 "Cannot get input Variable DDX, variable name = %s",
                 ctx.InputName("DDX")));

    // extract out(input), dout(output)
    auto out_var = ctx.InputVar("Out");
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "Cannot get input Variable Out, variable name = %s",
                     ctx.InputName("Out")));
    auto dout_var = ctx.OutputVar("DOut");
    Out = ctx.Input<framework::Tensor>("Out");
    if (dout_var) {
      dOut = ctx.Output<framework::Tensor>("DOut");
    }

    // extract dx(input)
    auto dx_var = ctx.InputVar("DX");
    PADDLE_ENFORCE_NOT_NULL(
        dx_var, platform::errors::NotFound(
                    "Cannot get input Variable DX, variable name = %s",
                    ctx.InputName("DX")));
    if (dx_var) {
      dX = ctx.Input<framework::Tensor>("DX");
    }

    if (dOut) dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());

    auto& place = ctx.template device_context<DeviceContext>();

    Functor functor;
    functor(place, Out, ddX, ddOut, dOut, dX);
  }
};

template <typename DeviceContext, typename Functor>
class PowKernel : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;

  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* X = nullptr;
    framework::Tensor* Out = nullptr;
    ExtractActivationTensor(context, &X, &Out);
    Out->mutable_data<T>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "Pow"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "Pow"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    // get FactorTensor
    auto* factor_tensor = context.HasInput("FactorTensor")
                              ? context.Input<framework::Tensor>("FactorTensor")
                              : nullptr;
    if (factor_tensor) {
      auto* factor_data = factor_tensor->data<float>();
      framework::Tensor cpu_factor_tensor;
      if (platform::is_gpu_place(factor_tensor->place())) {
        framework::TensorCopySync(*factor_tensor, platform::CPUPlace(),
                                  &cpu_factor_tensor);
        factor_data = cpu_factor_tensor.data<float>();
      }
      auto factor =
          std::vector<float>(factor_data, factor_data + factor_tensor->numel());
      PADDLE_ENFORCE_EQ(
          factor.size(), 1,
          platform::errors::InvalidArgument(
              "The shape of factor(tensor) must be [1] rather than %d",
              factor.size()));
      for (auto& attr : attrs) {
        *attr.second = factor[0];
      }
    }
    functor(*place, x, out);
  }
};

template <typename DeviceContext, typename Functor>
class PowGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor *X, *Out, *dOut;
    framework::Tensor* dX = nullptr;
    X = Out = dOut = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(context, &X, &Out, &dOut,
                                                    &dX);
    dX->mutable_data<T>(context.GetPlace());
    auto dout = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "PowGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "PowGrad"));
    auto dx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Output", "X@GRAD", "PowGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "PowGrad"));
    auto* place =
        context.template device_context<DeviceContext>().eigen_device();
    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    // get FactorTensor
    auto* factor_tensor =
        context.HasInput("FactorTensor")
            ? context.Input<framework::LoDTensor>("FactorTensor")
            : nullptr;
    if (factor_tensor) {
      auto* factor_data = factor_tensor->data<float>();
      framework::Tensor cpu_factor_tensor;
      if (platform::is_gpu_place(factor_tensor->place())) {
        framework::TensorCopySync(*factor_tensor, platform::CPUPlace(),
                                  &cpu_factor_tensor);
        factor_data = cpu_factor_tensor.data<float>();
      }
      auto factor =
          std::vector<float>(factor_data, factor_data + factor_tensor->numel());
      PADDLE_ENFORCE_EQ(
          factor.size(), 1,
          platform::errors::InvalidArgument(
              "The shape of factor(tensor) must be [1] rather than %d",
              factor.size()));
      for (auto& attr : attrs) {
        *attr.second = factor[0];
      }
    }
    functor(*place, x, out, dout, dx);
  }
};

template <typename DeviceContext, typename T>
class LogitGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto eps = context.Attr<float>("eps");
    dx->mutable_data<T>(dout->place());

    auto eigen_x = framework::EigenVector<T>::Flatten(*x);
    auto eigen_dout = framework::EigenVector<T>::Flatten(*dout);
    auto eigen_dx = framework::EigenVector<T>::Flatten(*dx);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto eigen_p = framework::EigenVector<T>::Flatten(*x);

    LogitGradFunctor<T> functor;
    functor(place, eigen_x, eigen_dout, eigen_dx, eigen_p, eps);
  }
};

template <typename T>
struct LogGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev, const framework::Tensor* X,
                  const framework::Tensor* ddX, framework::Tensor* ddOut,
                  const framework::Tensor* dOut, framework::Tensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "LogGradGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "LogGradGrad"));
    // ddout = ddx / x; dx = -(dout / x) * (ddx / x)
    // calculate dx first, so ddout can inplace ddx
    if (dX) {
      auto dout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Output", "DOut", "LogGradGrad"));
      auto dx = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dX, "Output", "DX", "LogGradGrad"));
      dx.device(*d) = dout * static_cast<T>(-1) * ddx / (x * x);
    }
    if (ddOut) {
      auto ddout = framework::EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "LogGradGrad"));
      ddout.device(*d) = ddx * static_cast<T>(1) / x;
    }
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_ACTIVATION_OP(__macro)                                      \
  __macro(logsigmoid, LogSigmoid, LogSigmoidFunctor, LogSigmoidGradFunctor); \
  __macro(ceil, Ceil, CeilFunctor, ZeroGradFunctor);                         \
  __macro(floor, Floor, FloorFunctor, ZeroGradFunctor);                      \
  __macro(round, Round, RoundFunctor, ZeroGradFunctor);                      \
  __macro(reciprocal, Reciprocal, ReciprocalFunctor, ReciprocalGradFunctor); \
  __macro(log1p, Log1p, Log1pFunctor, Log1pGradFunctor);                     \
  __macro(log2, Log2, Log2Functor, Log2GradFunctor);                         \
  __macro(log10, Log10, Log10Functor, Log10GradFunctor);                     \
  __macro(soft_relu, SoftRelu, SoftReluFunctor, SoftReluGradFunctor);        \
  __macro(stanh, STanh, STanhFunctor, STanhGradFunctor);                     \
  __macro(softplus, Softplus, SoftplusFunctor, SoftplusGradFunctor);         \
  __macro(softsign, Softsign, SoftsignFunctor, SoftsignGradFunctor);         \
  __macro(relu6, Relu6, Relu6Functor, Relu6GradFunctor);                     \
  __macro(hard_sigmoid, HardSigmoid, HardSigmoidFunctor,                     \
          HardSigmoidGradFunctor);                                           \
  __macro(swish, Swish, SwishFunctor, SwishGradFunctor);                     \
  __macro(mish, Mish, MishFunctor, MishGradFunctor);                         \
  __macro(hard_swish, HardSwish, HardSwishFunctor, HardSwishGradFunctor);
