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

template <typename T>
struct LogitFunctor {
  template <typename Device, typename X, typename Out>
  void operator()(Device d, X x, Out out, float eps) const {
      // logit(x) = ln(x/(1-x))
      // auto tmp_x = (x.cwiseMin(static_cast<T>(1.0f - eps))).cwiseMax(eps);
      auto tmp_x = (x.cwiseMin(static_cast<T>(1.0f - eps))).cwiseMax(static_cast<T>(eps));
#if defined(PADDLE_WITH_MKLML) && !defined(_WIN32) && !defined(__APPLE__) && \
    !defined(__OSX__) && !defined(PADDLE_WITH_CUDA) &&                       \
    !defined(PADDLE_WITH_HIP)
      auto x_data = tmp_x.data();
      auto out_data = tmp_x.data();
      int n = std::min(tmp_x.size(), out.size());

      //std::memset(out_data, 0, n * sizeof(T));
      math::CBlas<T>::SCAL(n, -1, x_data, 1);
      for (int i = 0; i < n; ++i) {
        x_data[i] += static_cast<T>(1);
      }
      math::CBlas<T>::VDIV(n, out_data, x_data, out_data);

      for (int i = 0; i < n; ++i) {
        out_data[i] = std::log(out_data[i]);
      }
#else
      /*
      if (std::is_same<T, platform::float16>::value) {
        VLOG(4) << "cast from float16 to float before computing";
        auto casted_x = x.template cast<float>();
        out.device(d) = (casted_x/(static_cast<T>(1)-casted_x)).log().template cast<T>();
      } else {
        out.device(d) = (x/(static_cast<T>(1)-x)).log();
      }
      */
       out.device(d) = (tmp_x/(static_cast<T>(1)-tmp_x)).log();
#endif
    
  }
};

template <typename T>
struct LogitGradFunctor {
  template <typename Device, typename X, typename dOut, typename dX, typename P>
  void operator()(Device d, X x, dOut dout, dX dx, P p, float eps) const {
    //logit(x)' = 1/(x*(1-x))
    dx.device(d) = (x < static_cast<T>(eps) || x > static_cast<T>(1.0 - eps))
        .select(dout * (static_cast<T>(1)/((static_cast<T>(1) - x) * x)), p.constant(static_cast<T>(0)));
  }
};

template <typename DeviceContext, typename T>
class LogitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    auto eps = context.Attr<float>("eps");
    out->mutable_data<T>(in->place());

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    LogitFunctor<T> functor;
    functor(place, eigen_in, eigen_out, eps);
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

}  // namespace operators
}  // namespace paddle
