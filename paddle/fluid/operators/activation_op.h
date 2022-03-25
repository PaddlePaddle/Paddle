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
USE_PHI_FUNCTOR(Sigmoid)
USE_PHI_DOUBLE_GRAD_FUNCTOR(Sigmoid)
USE_PHI_TRIPLE_GRAD_FUNCTOR(Sigmoid)
USE_PHI_FUNCTOR(LogSigmoid)
USE_PHI_FUNCTOR(HardSigmoid)
USE_PHI_FUNCTOR(Log)
USE_PHI_DOUBLE_GRAD_FUNCTOR(Log)
USE_PHI_FUNCTOR(Log2)
USE_PHI_FUNCTOR(Log10)
USE_PHI_FUNCTOR(Log1p)
USE_PHI_FUNCTOR(Swish)
USE_PHI_FUNCTOR(HardSwish)
USE_PHI_FUNCTOR(Pow)
USE_PHI_FUNCTOR(Exp)
USE_PHI_FUNCTOR(Expm1)
USE_PHI_FUNCTOR(Mish)
USE_PHI_FUNCTOR(STanh)
USE_PHI_FUNCTOR(Reciprocal)
USE_PHI_FUNCTOR(Square)
USE_PHI_FUNCTOR(Sqrt)
USE_PHI_FUNCTOR(Rsqrt)
USE_PHI_FUNCTOR(Softplus)

template <typename T>
using ELUGradNegativeAlphaFunctor = phi::funcs::ELUGradNegativeAlphaFunctor<T>;

template <typename T>
using RoundFunctor = phi::funcs::RoundFunctor<T>;

template <typename T>
using FloorFunctor = phi::funcs::FloorFunctor<T>;

template <typename T>
using CeilFunctor = phi::funcs::CeilFunctor<T>;

template <typename T>
using ZeroGradFunctor = phi::funcs::ZeroGradFunctor<T>;

template <typename T>
using ELUGradNegativeAlphaFunctor = phi::funcs::ELUGradNegativeAlphaFunctor<T>;

// relu(x) = max(x, 0)

template <typename T>
using ReluCPUFunctor = phi::funcs::ReluCPUFunctor<T>;
template <typename T>
using ReluGradFunctor = phi::funcs::ReluGradFunctor<T>;

template <typename T>
using ReluGradGradFunctor = phi::funcs::ReluGradGradFunctor<T>;

template <typename T>
using ReluCUDAFunctor = phi::funcs::ReluCUDAFunctor<T>;

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

template <typename T>
struct SoftsignFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x / (static_cast<T>(1) + x.abs());
  }
};

// d(softsign(x))/dx = 1 / (1 + |x|)^2
// Taken from https://en.wikipedia.org/wiki/Activation_function

template <typename T>
struct SoftsignGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        dout * (static_cast<T>(1) / (static_cast<T>(1) + x.abs()).square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
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

}  // namespace operators
}  // namespace paddle

#define FOR_EACH_ACTIVATION_OP(__macro)                               \
  __macro(soft_relu, SoftRelu, SoftReluFunctor, SoftReluGradFunctor); \
  __macro(softsign, Softsign, SoftsignFunctor, SoftsignGradFunctor);  \
  __macro(relu6, Relu6, Relu6Functor, Relu6GradFunctor);
