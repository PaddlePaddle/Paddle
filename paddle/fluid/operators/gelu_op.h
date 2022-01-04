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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <algorithm>
#include <cmath>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/float16.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

#define GELU_CONSTANT 0.044715

template <typename T>
struct GeluFunctor {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out, bool approximate) const {
    if (approximate) {
      // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
      if (std::is_same<T, platform::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto temp =
            (static_cast<float>(M_2_SQRTPI * M_SQRT1_2) *
             (casted_x + static_cast<float>(GELU_CONSTANT) * casted_x.cube()))
                .tanh();
        out.device(d) = (casted_x * static_cast<float>(0.5) *
                         (static_cast<float>(1) + temp))
                            .template cast<T>();
      } else {
        auto temp = (static_cast<T>(M_2_SQRTPI * M_SQRT1_2) *
                     (x + static_cast<T>(GELU_CONSTANT) * x.cube()))
                        .tanh();
        out.device(d) = x * static_cast<T>(0.5) * (static_cast<T>(1) + temp);
      }
    } else {
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__) && !defined(PADDLE_WITH_CUDA) &&                       \
    !defined(PADDLE_WITH_HIP)
      auto x_data = x.data();
      auto out_data = out.data();
      int n = std::min(x.size(), out.size());

      std::memset(out_data, 0, n * sizeof(T));
      math::CBlas<T>::AXPY(n, static_cast<T>(M_SQRT1_2), x_data, 1, out_data,
                           1);
      math::CBlas<T>::VMERF(n, out_data, out_data, VML_LA);
      for (int i = 0; i < n; i++) {
        out_data[i] += static_cast<T>(1);
      }
      math::CBlas<T>::VMUL(n, x_data, out_data, out_data);
      for (int i = 0; i < n; i++) {
        out_data[i] *= static_cast<T>(0.5);
      }
#else
      // gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
      if (std::is_same<T, platform::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto temp = (casted_x * static_cast<float>(M_SQRT1_2)).erf();
        out.device(d) = (casted_x * static_cast<float>(0.5) *
                         (static_cast<float>(1) + temp))
                            .template cast<T>();
      } else {
        auto temp = (x * static_cast<T>(M_SQRT1_2)).erf();
        out.device(d) = x * static_cast<T>(0.5) * (static_cast<T>(1) + temp);
      }
#endif
    }
  }
};

template <typename T>
struct GeluGradFunctor {
  template <typename Device, typename X, typename dOut, typename dX>
  void operator()(Device d, X x, dOut dout, dX dx, bool approximate) const {
    if (approximate) {
      if (std::is_same<T, platform::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto casted_dout = dout.template cast<float>();

        const float kAlpha = static_cast<float>(M_2_SQRTPI * M_SQRT1_2);
        const float kBeta =
            kAlpha * static_cast<float>(GELU_CONSTANT) * static_cast<float>(3);
        const auto y =
            (kAlpha *
             ((static_cast<float>(GELU_CONSTANT) * casted_x.cube()) + casted_x))
                .tanh();
        dx.device(d) = (static_cast<float>(0.5) * casted_dout *
                        (static_cast<float>(1) + y +
                         (casted_x - casted_x * y.square()) *
                             (kAlpha + kBeta * casted_x.square())))
                           .template cast<T>();
      } else {
        const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
        const T kBeta =
            kAlpha * static_cast<T>(GELU_CONSTANT) * static_cast<T>(3);
        const auto y =
            (kAlpha * ((static_cast<T>(GELU_CONSTANT) * x.cube()) + x)).tanh();
        dx.device(d) = static_cast<T>(0.5) * dout *
                       (static_cast<T>(1) + y +
                        (x - x * y.square()) * (kAlpha + kBeta * x.square()));
      }
    } else {
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__) && !defined(PADDLE_WITH_CUDA) &&                       \
    !defined(PADDLE_WITH_HIP)
      auto x_data = x.data();
      auto dx_data = dx.data();
      auto dout_data = dout.data();
      int n = std::min(x.size(), dx.size());

      auto first = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(first, 0, n * sizeof(T));
      auto second = static_cast<T*>(std::malloc(n * sizeof(T)));
      std::memset(second, 0, n * sizeof(T));

      // first = (0.5 * (1 + erf(x / sqrt(2))))
      math::CBlas<T>::AXPY(n, static_cast<T>(M_SQRT1_2), x_data, 1, first, 1);
      math::CBlas<T>::VMERF(n, first, first, VML_LA);
      for (int i = 0; i < n; i++) {
        first[i] += static_cast<T>(1);
      }
      math::CBlas<T>::SCAL(n, static_cast<T>(0.5), first, 1);

      // second = (0.5 * 2/sqrt(pi) * 1/sqrt(2) * x * exp(-0.5 * x^2))
      math::CBlas<T>::VSQUARE(n, x_data, second);
      math::CBlas<T>::SCAL(n, -static_cast<T>(0.5), second, 1);
      math::CBlas<T>::VEXP(n, second, second);
      math::CBlas<T>::VMUL(n, x_data, second, second);
      math::CBlas<T>::SCAL(n, static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2),
                           second, 1);

      // dx = dout * (first + second);
      math::CBlas<T>::VADD(n, first, second, first);
      math::CBlas<T>::VMUL(n, dout_data, first, dx_data);

      std::free(first);
      std::free(second);
#else
      // gelu_grad(x) = dout * 0.5 * (1 + erf(x / sqrt(2)) + x * sqrt(2 / pi) *
      // exp(- x^2 / 2)
      if (std::is_same<T, platform::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        auto casted_dout = dout.template cast<float>();
        auto first = static_cast<float>(0.5) *
                     (static_cast<float>(1) +
                      ((casted_x * static_cast<float>(M_SQRT1_2)).erf()));
        auto second = static_cast<float>(0.5 * M_2_SQRTPI * M_SQRT1_2) *
                      casted_x *
                      (-static_cast<float>(0.5) * casted_x.square()).exp();
        dx.device(d) = (casted_dout * (first + second)).template cast<T>();
      } else {
        auto first =
            static_cast<T>(0.5) *
            (static_cast<T>(1) + ((x * static_cast<T>(M_SQRT1_2)).erf()));

        auto second = static_cast<T>(0.5 * M_2_SQRTPI * M_SQRT1_2) * x *
                      (-static_cast<T>(0.5) * x.square()).exp();
        dx.device(d) = dout * (first + second);
      }
#endif
    }
  }
};

template <typename DeviceContext, typename T>
class GeluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    auto approximate = context.Attr<bool>("approximate");
    out->mutable_data<T>(in->place());

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    GeluFunctor<T> functor;
    functor(place, eigen_in, eigen_out, approximate);
  }
};

template <typename DeviceContext, typename T>
class GeluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto approximate = context.Attr<bool>("approximate");
    dx->mutable_data<T>(dout->place());

    auto eigen_x = framework::EigenVector<T>::Flatten(*x);
    auto eigen_dout = framework::EigenVector<T>::Flatten(*dout);
    auto eigen_dx = framework::EigenVector<T>::Flatten(*dx);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    GeluGradFunctor<T> functor;
    functor(place, eigen_x, eigen_dout, eigen_dx, approximate);
  }
};

}  // namespace operators
}  // namespace paddle
