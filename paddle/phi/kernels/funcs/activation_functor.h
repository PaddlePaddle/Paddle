// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
namespace funcs {
enum ActBwdOpFwdDeps {
  kNoDeps = 0x00,  // Do not need any forward input/output
  kDepX = 0x01,    // Only need forward input X
  kDepOut = 0x02,  // Only need forward output Out
};

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char*, float*>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

template <typename T>
struct Sine {
  HOSTDEVICE T operator()(const T& val) const { return sin(val); }
};

template <>
struct Sine<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(sin(static_cast<float>(val)));
  }
};

template <typename T>
struct Cosine {
  HOSTDEVICE T operator()(const T& val) const { return cos(val); }
};

template <>
struct Cosine<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(cos(static_cast<float>(val)));
  }
};

// sine'(x) = cos(x)
template <typename T>
struct SinGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * x.unaryExpr(Cosine<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// sine(x) = sin(x)
template <typename T>
struct SinFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Sine<T>());
  }
};

// cosine'(x) = -sin(x)
template <typename T>
struct CosGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = -dout * x.unaryExpr(Sine<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// cosine(x) = cos(x)
template <typename T>
struct CosFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Cosine<T>());
  }
};

template <typename T>
struct Tangent {
  HOSTDEVICE T operator()(const T& val) const { return tan(val); }
};

template <>
struct Tangent<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(tan(static_cast<float>(val)));
  }
};

// Tangent'(x) = -Tangent(x)
template <typename T>
struct TanGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout / x.unaryExpr(Cosine<T>()).square();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// Tangent(x) = tan(x)
template <typename T>
struct TanFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Tangent<T>());
  }
};

template <typename T>
struct Sinh {
  HOSTDEVICE T operator()(const T& val) const { return sinh(val); }
};

template <>
struct Sinh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(sinhf(static_cast<float>(val)));
  }
};

template <typename T>
struct Cosh {
  HOSTDEVICE T operator()(const T& val) const { return cosh(val); }
};

template <>
struct Cosh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(coshf(static_cast<float>(val)));
  }
};

// sinh(x) = sinh(x)
template <typename T>
struct SinhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Sinh<T>());
  }
};

// cosh(x) = cosh(x)
template <typename T>
struct CoshFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Cosh<T>());
  }
};

// sinh'(x) = cosh(x)
template <typename T>
struct SinhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * x.unaryExpr(Cosh<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// cosh'(x) = sinh(x)
template <typename T>
struct CoshGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * x.unaryExpr(Sinh<T>());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Acos {
  HOSTDEVICE T operator()(const T& val) const { return acos(val); }
};

template <>
struct Acos<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(acos(static_cast<float>(val)));
  }
};

// Acos(x) = acos(x)
template <typename T>
struct AcosFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Acos<T>());
  }
};

// acos'(x) = -1/sqrt(1-x^2)
template <typename T>
struct AcosGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        -dout * static_cast<T>(1) / (static_cast<T>(1) - x.square()).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Asin {
  HOSTDEVICE T operator()(const T& val) const { return asin(val); }
};

template <>
struct Asin<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(asin(static_cast<float>(val)));
  }
};

// Asin(x) = asin(x)
template <typename T>
struct AsinFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Asin<T>());
  }
};

// asin'(x) = 1/sqrt(1-x^2)
template <typename T>
struct AsinGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<T>(1) / (static_cast<T>(1) - x.square()).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Atan {
  HOSTDEVICE T operator()(const T& val) const { return atan(val); }
};

template <>
struct Atan<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(atan(static_cast<float>(val)));
  }
};

// Atan(x) = atan(x)
template <typename T>
struct AtanFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Atan<T>());
  }
};

// atan'(x) =  1 / (1 + x^2)
template <typename T>
struct AtanGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (static_cast<T>(1) + x.square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Acosh {
  HOSTDEVICE T operator()(const T& val) const { return acosh(val); }
};

template <>
struct Acosh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(acosh(static_cast<float>(val)));
  }
};

// Acosh(x) = acosh(x)
template <typename T>
struct AcoshFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Acosh<T>());
  }
};

// acosh'(x) =  1/sqrt(x^2 - 1)
template <typename T>
struct AcoshGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<T>(1) / (x * x - static_cast<T>(1)).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Asinh {
  HOSTDEVICE T operator()(const T& val) const { return asinh(val); }
};

template <>
struct Asinh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(asinh(static_cast<float>(val)));
  }
};

// Asinh(x) = asinh(x)
template <typename T>
struct AsinhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Asinh<T>());
  }
};

// asinh'(x) =  1/sqrt(x^2 + 1)
template <typename T>
struct AsinhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) =
        dout * static_cast<T>(1) / (x.square() + static_cast<T>(1)).sqrt();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

template <typename T>
struct Atanh {
  HOSTDEVICE T operator()(const T& val) const { return atanh(val); }
};

template <>
struct Atanh<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(const dtype::float16& val) const {
    return dtype::float16(atanh(static_cast<float>(val)));
  }
};

// Atanh(x) = atanh(x)
template <typename T>
struct AtanhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr(Atanh<T>());
  }
};

// atanh'(x) =  1/(1 - x^2)
template <typename T>
struct AtanhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * static_cast<T>(1) / (static_cast<T>(1) - x.square());
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepX; }
};

// relu(x) = max(x, 0)
template <typename T>
struct ReluCPUFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.unaryExpr([] HOSTDEVICE(T v) {
      return v > static_cast<T>(0) ? v : static_cast<T>(0);
    });
  }
};

template <typename T>
struct ReluCUDAFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.cwiseMax(static_cast<T>(0));
  }
};

template <typename T>
struct ReluGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (out > static_cast<T>(0)).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct ReluGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  DenseTensor* dOut,
                  DenseTensor* dX) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "ReluGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Output", "Out", "ReluGradGrad"));
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "ReluGradGrad"));
      ddout.device(*d) = ddx * (out > static_cast<T>(0)).template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
template <typename T>
struct TanhFunctor : public BaseActivationFunctor<T> {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) = x.tanh();
  }
};

template <typename T>
struct TanhGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout * (static_cast<T>(1) - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct TanhGradGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  DenseTensor* dOutNew,
                  DenseTensor* ddOut) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "TanhGradGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "TanhGradGrad"));
    // tanh grad grad : ddout = (1 - out^2) * ddx, dout = - (dout_old * 2 * out
    // * ddx)
    if (dOutNew) {
      auto dout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOut, "Input", "DOut", "TanhGradGrad"));
      auto dout_new = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(dOutNew, "Output", "DOutNew", "TanhGradGrad"));
      dout_new.device(*d) =
          static_cast<T>(-1) * dout * static_cast<T>(2) * out * ddx;
    }
    if (ddOut) {
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DDOut", "TanhGradGrad"));
      ddout.device(*d) = (static_cast<T>(1) - out * out) * ddx;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};
/*
    Out
    DOut                            D_Dout
    DDx     -> TanhTripleGrad ->    D_DDx
    D_DDout                         d_OutNew
    D_Dout_new

    D_Dout = (-2) * Out * DDx * D_Dout_new
    D_DDx = (1-Out^2)*D_DDout + (-2) * Out * DOut * D_Dout_new
    D_OutNew = (-2) * Out * DDx * D_DDout + (-2) * DOut * DDx * D_Dout_new

    Out, DDX, DOut, D_DDOut, D_DOut_New   // input
    D_OutNew, D_DOut, D_DDx               // output
*/
template <typename T>
struct TanhTripleGradFunctor : public BaseActivationFunctor<T> {
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  const DenseTensor* dOut,
                  const DenseTensor* d_DDOut,
                  const DenseTensor* d_dOut_New,
                  DenseTensor* d_d_Out,
                  DenseTensor* d_Out_New,
                  DenseTensor* d_DDx) const {
    auto* d = dev.eigen_device();
    auto ddx = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(ddX, "Input", "DDX", "TanhTripleGrad"));
    auto out = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(Out, "Input", "Out", "TanhTripleGrad"));
    auto dout = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(dOut, "Input", "DOut", "TanhTripleGrad"));
    auto d_ddOut = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(d_DDOut, "Input", "D_DDOut", "TanhTripleGrad"));
    auto d_dOutNew = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(d_dOut_New, "Input", "D_DOut_New", "TanhTripleGrad"));

    if (d_Out_New) {
      auto d_OutNew = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_Out_New, "Output", "D_OutNew", "TanhTripleGrad"));
      d_OutNew.device(*d) = (static_cast<T>(-2) * out * ddx * d_ddOut) -
                            (static_cast<T>(2) * dout * ddx * d_dOutNew);
    }
    if (d_d_Out) {
      auto d_dOut = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_d_Out, "Output", "D_DOut", "TanhTripleGrad"));
      d_dOut.device(*d) = static_cast<T>(-2) * out * ddx * d_dOutNew;
    }
    if (d_DDx) {
      auto d_ddx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(d_DDx, "Output", "D_DDx", "TanhTripleGrad"));
      d_ddx.device(*d) = (static_cast<T>(1) - (out * out)) * d_ddOut -
                         static_cast<T>(2) * out * dout * d_dOutNew;
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
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

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    out.device(d) =
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
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    dx.device(d) = dout *
                   ((x > static_cast<T>(t_min)) * (x < static_cast<T>(t_max)))
                       .template cast<T>();
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
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
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
struct LeakyReluFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    if (alpha < 1.f) {
      out.device(d) = x.cwiseMax(static_cast<T>(alpha) * x);
    } else {
      out.device(d) = x.cwiseMin(static_cast<T>(alpha) * x);
    }
  }
};

template <typename T>
struct LeakyReluGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto temp1 =
        static_cast<T>(alpha) * (x < static_cast<T>(0)).template cast<T>();
    auto temp2 = (x >= static_cast<T>(0)).template cast<T>();
    dx.device(d) = dout * (temp1 + temp2).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct LeakyReluGradGradFunctor : public BaseActivationFunctor<T> {
  float alpha;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }
  template <typename Device>
  void operator()(const Device& dev,
                  const DenseTensor* X,
                  const DenseTensor* Out,
                  const DenseTensor* ddX,
                  DenseTensor* ddOut,
                  DenseTensor* dOut,
                  DenseTensor* dX) const {
    if (ddOut) {
      auto* d = dev.eigen_device();
      auto ddx = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddX, "Input", "DDX", "LeakyReluGradGrad"));
      auto x = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(X, "Input", "X", "LeakyReluGradGrad"));
      auto ddout = EigenVector<T>::Flatten(
          GET_DATA_SAFELY(ddOut, "Output", "DOut", "LeakyReluGradGrad"));
      ddout.device(*d) =
          ddx *
          ((x > static_cast<T>(0)).template cast<T>() +
           static_cast<T>(alpha) * (x <= static_cast<T>(0)).template cast<T>())
              .template cast<T>();
    }
  }
  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct ThresholdedReluFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out) const {
    auto th = static_cast<T>(threshold);
    out.device(d) = (x > th).template cast<T>() * x;
  }
};

template <typename T>
struct ThresholdedReluGradFunctor : public BaseActivationFunctor<T> {
  float threshold;
  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x, Out out, dOut dout, dX dx) const {
    auto th = static_cast<T>(threshold);
    dx.device(d) = dout * (x > th).template cast<T>();
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
template <typename T>
struct CudaReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // relu(x) = max(x, 0)
  __device__ __forceinline__ T operator()(const T x) const {
    return x > zero ? x : zero;
  }
};

template <typename T>
struct CudaReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);

  // dx = dout * (out > 0)
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return out > zero ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaCosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // cos(x) = cos(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(cos(x));
  }
};

template <typename T>
struct CudaCosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * (-sin(x))
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(-dout * sin(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // sin(x) = sin(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(sin(x));
  }
};

template <typename T>
struct CudaSinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * cos(x)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * cos(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaTanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // tan(x) = tan(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(tan(x));
  }
};

template <typename T>
struct CudaTanGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout / cos(x)^2
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout / (cos(x) * cos(x)));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAsinFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // asin(x) = asin(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(asin(x));
  }
};

template <typename T>
struct CudaAsinGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout / sqrt(1 - x^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAcosFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // acos(x) = acos(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(acos(x));
  }
};

template <typename T>
struct CudaAcosGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = -dout / sqrt(1 - x^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(-dout / sqrt(one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaCoshFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // cosh(x) = cosh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(cosh(x));
  }
};

template <typename T>
struct CudaCoshGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * sinh(x)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * sinh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaSinhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // sinh(x) = sinh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(sinh(x));
  }
};

template <typename T>
struct CudaSinhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // dx = dout * cosh(x)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * cosh(x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAcoshFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // Acosh(x) = acosh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(acosh(x));
  }
};

template <typename T>
struct CudaAcoshGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  // dx = dout * 1 / sqrt(x^2 - 1)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * one / sqrt(x * x - one));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAsinhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // Asinh(x) = asinh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(asinh(x));
  }
};

template <typename T>
struct CudaAsinhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);

  // dx = dout * 1/sqrt(x^2 + 1)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * one / sqrt(x * x + one));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAtanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // Atanh(x) = atanh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(atanh(x));
  }
};

template <typename T>
struct CudaAtanhGradFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  // dx = dout * 1/(1- x^2)
  __device__ __forceinline__ T operator()(const T arg_dout,
                                          const T arg_x) const {
    MPType dout = static_cast<MPType>(arg_dout);
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(dout * one / (one - x * x));
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaAtanFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // atan(x) = atan(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(atan(x));
  }
};

template <typename T>
struct CudaAtanGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout / (1 + x^2)
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return dout / (one + x * x);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaTanhFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  // tanh(x) = tanh(x)
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(tanh(x));
  }
};

template <typename T>
struct CudaTanhGradFunctor : public BaseActivationFunctor<T> {
  T one = static_cast<T>(1.0f);

  // dx = dout * (1 - out^2)
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    return dout * (one - out * out);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaBReluFunctor : public BaseActivationFunctor<T> {
  float t_min;
  float t_max;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  // brelu(x) = min(max(x, t_min), t_max)
  __device__ __forceinline__ T operator()(const T x) const {
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    T temp_max = x > t_min_cast ? x : t_min_cast;
    T temp_min = temp_max < t_max_cast ? temp_max : t_max_cast;
    return temp_min;
  }
};

template <typename T>
struct CudaBReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float t_min;
  float t_max;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"t_min", &t_min}, {"t_max", &t_max}};
  }

  // dx = (x > t_min && x < t_max) ? dout : 0
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    T t_min_cast = static_cast<T>(t_min);
    T t_max_cast = static_cast<T>(t_max);
    return (x > t_min_cast && x < t_max_cast) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaRelu6Functor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // relu6(x) = min(max(0, x), 6)
  __device__ __forceinline__ T operator()(const T x) const {
    T t = static_cast<T>(threshold);
    return x <= zero ? zero : (x < t ? x : t);
  }
};

template <typename T>
struct CudaRelu6GradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = (out > 0 && out < t) ? dout : 0
  __device__ __forceinline__ T operator()(const T dout, const T out) const {
    T t = static_cast<T>(threshold);
    return (out > zero && out < t) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() {
    return ActBwdOpFwdDeps::kDepOut;
  }
};

template <typename T>
struct CudaThresholdedReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // thresholded_relu(x) = x > threshold ? x : 0
  __device__ __forceinline__ T operator()(const T x) const {
    return x > static_cast<T>(threshold) ? x : zero;
  }
};

template <typename T>
struct CudaThresholdedReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float threshold;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"threshold", &threshold}};
  }

  // dx = x > threshold ? dout : 0
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return x > static_cast<T>(threshold) ? dout : zero;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};

template <typename T>
struct CudaLeakyReluFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // leakyrelu(x) = x > 0 ? x : alpha * x
  __device__ __forceinline__ T operator()(const T x) const {
    return x > zero ? x : static_cast<T>(alpha) * x;
  }
};

template <typename T>
struct CudaLeakyReluGradFunctor : public BaseActivationFunctor<T> {
  T zero = static_cast<T>(0.0f);
  float alpha;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"alpha", &alpha}};
  }

  // dx = dout * (x > 0 ? 1 : alpha)
  __device__ __forceinline__ T operator()(const T dout, const T x) const {
    return x > zero ? dout : static_cast<T>(alpha) * dout;
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return ActBwdOpFwdDeps::kDepX; }
};
#endif

}  // namespace funcs
}  // namespace phi
