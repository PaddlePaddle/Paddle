// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using framework::To32BitIndex;

template <typename T>
struct Atan2Functor {
  Atan2Functor(const T* x1, const T* x2, T* out, int64_t numel)
      : x1_(x1), x2_(x2), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = ::atan2(x1_[idx], x2_[idx]);
  }

  const T* x1_;
  const T* x2_;
  T* out_;
  int64_t numel_;
};

template <>
struct Atan2Functor<float> {
  Atan2Functor(const float* x1, const float* x2, float* out, int64_t numel)
      : x1_(x1), x2_(x2), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = ::atan2f(x1_[idx], x2_[idx]);
  }

  const float* x1_;
  const float* x2_;
  float* out_;
  int64_t numel_;
};
template <>
struct Atan2Functor<platform::float16> {
  Atan2Functor(const platform::float16* x1, const platform::float16* x2,
               platform::float16* out, int64_t numel)
      : x1_(x1), x2_(x2), out_(out), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    out_[idx] = static_cast<platform::float16>(
        ::atan2f(static_cast<float>(x1_[idx]), static_cast<float>(x2_[idx])));
  }

  const platform::float16* x1_;
  const platform::float16* x2_;
  platform::float16* out_;
  int64_t numel_;
};

// dx1 = dout * x2 / ((x1)^2 + (x2)^2)
// dx2 = - dout * x1 / ((x1)^2 + (x2)^2)
template <typename T>
struct Atan2GradFunctor {
  Atan2GradFunctor(const T* x1, const T* x2, const T* dout, T* dx1, T* dx2,
                   int64_t numel)
      : x1_(x1), x2_(x2), dout_(dout), dx1_(dx1), dx2_(dx2), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    auto x = x1_[idx] * x1_[idx] + x2_[idx] * x2_[idx];
    dx1_[idx] = dout_[idx] * x2_[idx] / x;
    dx2_[idx] = -dout_[idx] * x1_[idx] / x;
  }

  const T* x1_;
  const T* x2_;
  const T* dout_;
  T* dx1_;
  T* dx2_;
  int64_t numel_;
};

template <>
struct Atan2GradFunctor<platform::float16> {
  Atan2GradFunctor(const platform::float16* x1, const platform::float16* x2,
                   const platform::float16* dout, platform::float16* dx1,
                   platform::float16* dx2, int64_t numel)
      : x1_(x1), x2_(x2), dout_(dout), dx1_(dx1), dx2_(dx2), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    float x1 = static_cast<float>(x1_[idx]);
    float x2 = static_cast<float>(x2_[idx]);
    float x = x1 * x1 + x2 * x2;
    dx1_[idx] =
        static_cast<platform::float16>(static_cast<float>(dout_[idx]) * x2 / x);
    dx2_[idx] = static_cast<platform::float16>(-static_cast<float>(dout_[idx]) *
                                               x1 / x);
  }

  const platform::float16* x1_;
  const platform::float16* x2_;
  const platform::float16* dout_;
  platform::float16* dx1_;
  platform::float16* dx2_;
  int64_t numel_;
};

template <typename DeviceContext, typename T>
class Atan2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* X1 = context.Input<Tensor>("X1");
    const Tensor* X2 = context.Input<Tensor>("X2");
    Tensor* Out = context.Output<Tensor>("Out");

    auto numel = X1->numel();
    auto x1 = X1->data<T>();
    auto x2 = X2->data<T>();
    auto out =
        Out->mutable_data<T>(context.GetPlace(), size_t(numel * sizeof(T)));
    auto& dev_ctx = context.template device_context<DeviceContext>();

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    Atan2Functor<T> functor(x1, x2, out, numel);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class Atan2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const {
    const Tensor* X1 = context.Input<Tensor>("X1");
    const Tensor* X2 = context.Input<Tensor>("X2");
    const Tensor* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* dX1 = context.Output<Tensor>(framework::GradVarName("X1"));
    Tensor* dX2 = context.Output<Tensor>(framework::GradVarName("X2"));

    auto numel = X1->numel();
    auto x1 = X1->data<T>();
    auto x2 = X2->data<T>();
    auto dout = dOut->data<T>();
    auto dx1 =
        dX1->mutable_data<T>(context.GetPlace(), size_t(numel * sizeof(T)));
    auto dx2 =
        dX2->mutable_data<T>(context.GetPlace(), size_t(numel * sizeof(T)));
    auto& dev_ctx = context.template device_context<DeviceContext>();

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    Atan2GradFunctor<T> functor(x1, x2, dout, dx1, dx2, numel);
    for_range(functor);
  }
};
}  // namespace operators
}  // namespace paddle
