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

#include <cmath>
#include <type_traits>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct PowFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
// TODO(wujionghao): A potential speed improvement is supporting different
// types in C++.
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    // On CUDAPlace, std::pow(3, 1) calls pow(float, float), and
    // it will return a float number like 2.99... , which floor to 2
    // when cast to int by default and it is wrong.
    // Use llrint to cast it to the nearest integer, which is 3.
    if (std::is_integral<T>::value) {
      return std::llrint(
          std::pow(static_cast<double>(a), static_cast<double>(b)));
    }
#endif
    return std::pow(a, b);
  }
};

template <typename DeviceContext, typename T>
class ElementwisePowKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::LoDTensor;
    auto* x = ctx.Input<Tensor>("X");
    PADDLE_ENFORCE_EQ(x != nullptr, true,
                      platform::errors::NotFound(
                          "Cannot get input Variable X, Variable name = %s",
                          ctx.InputName("X")));
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<PowFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          PowFunctor<T>(), z);
  }
};

template <typename T>
struct PowGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    if (std::is_integral<T>::value) {
      return dout * y *
             std::pow(static_cast<double>(x), static_cast<double>(y - 1));
    }
#endif
    return dout * y * std::pow(x, y - 1);
  }
};

template <typename T, typename Enable = void>
struct PowGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    if (std::is_integral<T>::value) {
      return dout * std::log(static_cast<double>(x)) *
             std::pow(static_cast<double>(x), static_cast<double>(y));
    }
#endif
    return dout * std::log(x) * std::pow(x, y);
  }
};

template <typename DeviceContext, typename T>
class ElementwisePowGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = dout;
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, PowGradDX<T>, PowGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, PowGradDX<T>(), PowGradDY<T>());
  }
};
}  // namespace operators
}  // namespace paddle
