/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
using EnableComplex =
    typename std::enable_if<std::is_same<T, platform::complex64>::value ||
                            std::is_same<T, platform::complex128>::value>::type;

template <typename T>
using DisableComplex = typename std::enable_if<
    !std::is_same<T, platform::complex64>::value &&
    !std::is_same<T, platform::complex128>::value>::type;

template <typename T, typename Enable = void>
struct AbsFunctor;

// abs(x) = |x|
template <typename T>
struct AbsFunctor<T, DisableComplex<T>> : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.abs();
  }
};

// complex: abs(x) = (x*x)^-1
template <typename T>
struct AbsFunctor<T, EnableComplex<T>> : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.abs().real();
  }
};

template <typename T, typename Enable = void>
struct AbsGradFunctor;

template <typename T>
struct AbsGradFunctor<T, DisableComplex<T>> : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * x.sign();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct AbsGradFunctor<T, EnableComplex<T>> : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out, typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (x / x.abs());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
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
  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename DeviceContext, typename Functor>
class AbsKernel : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;

  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* X = nullptr;
    framework::Tensor* Out = nullptr;
    ExtractActivationTensor(context, &X, &Out);
    Out->mutable_data<math::Real<T>>(context.GetPlace());

    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "Abs"));
    auto out = framework::EigenVector<math::Real<T>>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "Abs"));
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
class AbsGradKernel
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
        GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "AbsGrad"));
    auto out = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "AbsGrad"));
    auto dx = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dX, "Input", "X@GRAD", "AbsGrad"));
    auto x = framework::EigenVector<T>::Flatten(
        GET_DATA_SAFELY(X, "Input", "X", "AbsGrad"));
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

template <typename DeviceContext, typename Functor>
class AbsDoubleGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor *X, *Out, *ddX;
    X = Out = ddX = nullptr;
    framework::Tensor *ddOut, *dOut, *dX;
    ddOut = dOut = dX = nullptr;

    ExtractActivationDoubleGradTensor<Functor::FwdDeps()>(ctx, &X, &Out, &ddX,
                                                          &dX, &dOut, &ddOut);

    if (ddOut) ddOut->mutable_data<T>(ctx.GetPlace());
    if (dOut) dOut->mutable_data<T>(ctx.GetPlace());
    if (dX) dX->mutable_data<T>(Out->dims(), ctx.GetPlace());

    auto& place = ctx.template device_context<DeviceContext>();

    Functor functor;
    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = ctx.Attr<float>(attr.first);
    }
    functor(place, X, Out, ddX, ddOut, dOut, dX);
  }
};

}  // namespace operators
}  // namespace paddle
