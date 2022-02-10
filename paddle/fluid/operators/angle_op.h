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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "paddle/fluid/operators/math/complex_functors.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

namespace math {
template <typename T, typename Enable = void>
struct AngleFunctor;

// angel function for complex
template <typename T>
struct AngleFunctor<T, Complex<T, Real<T>>> {
  AngleFunctor(const T* input, Real<T>* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = arg(input_[idx]);
  }

  const T* input_;
  Real<T>* output_;
  int64_t numel_;
};

// angel function for real
template <typename T>
struct AngleFunctor<T, NoComplex<T, Real<T>>> {
  AngleFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = input_[idx] < static_cast<T>(0) ? M_PI : 0;
  }

  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct AngleGradFunctor;

// angle grad for complex
template <typename T>
struct AngleGradFunctor<T, Complex<T, Real<T>>> {
  AngleGradFunctor(const math::Real<T>* dout, const T* x, T* dx, int64_t numel)
      : dout_(dout), x_(x), dx_(dx), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == T(0)) {
      dx_[idx] = T(0);
    } else {
      const math::Real<T> r_square =
          x_[idx].real * x_[idx].real + x_[idx].imag * x_[idx].imag;
      dx_[idx] = T(-dout_[idx] * x_[idx].imag / r_square,
                   dout_[idx] * x_[idx].real / r_square);
    }
  }

  const math::Real<T>* dout_;
  const T* x_;
  T* dx_;
  int64_t numel_;
};

// angle grad for real
template <typename T>
struct AngleGradFunctor<T, NoComplex<T, Real<T>>> {
  AngleGradFunctor(const math::Real<T>* dout, const T* x, T* dx, int64_t numel)
      : dout_(dout), x_(x), dx_(dx), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const { dx_[idx] = 0; }

  const math::Real<T>* dout_;
  const T* x_;
  T* dx_;
  int64_t numel_;
};
}  // namespace math

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class AngleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    auto numel = x->numel();
    auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<math::Real<T>>(
        context.GetPlace(), size_t(x->numel() * sizeof(math::Real<T>)));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    math::AngleFunctor<T> functor(x_data, out_data, numel);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class AngleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    auto* dout_data = d_out->data<math::Real<T>>();
    auto* x_data = x->data<T>();
    auto* dx_data = d_x->mutable_data<T>(
        ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    math::AngleGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
    for_range(functor);
  }
};
}  // namespace operators
}  // namespace paddle
