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

#if defined(__NVCC__) || defined(__HIPCC__)
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

#endif

}  // namespace funcs
}  // namespace phi
