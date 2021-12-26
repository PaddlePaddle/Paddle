/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {

// functors to use with ElementwiseComputeEx
template <typename T>
struct RealAndImagToComplexFunctor {
  inline HOSTDEVICE platform::complex<T> operator()(const T& x, const T& y) {
    return platform::complex<T>(x, y);
  }
};

template <typename T>
struct ImagAndRealToComplexFunctor {
  inline HOSTDEVICE platform::complex<T> operator()(const T& y, const T& x) {
    return platform::complex<T>(x, y);
  }
};

template <typename T>
struct ComplexGradForRealFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y,
                                 const platform::complex<T> out,
                                 const platform::complex<T> dout) {
    return dout.real;
  }
};

template <typename T>
struct ComplexGradForImagFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y,
                                 const platform::complex<T> out,
                                 const platform::complex<T> dout) {
    return dout.imag;
  }
};

template <typename DeviceContext, typename T>
class ComplexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<framework::Tensor>("X");
    const auto* y = ctx.Input<framework::Tensor>("Y");
    auto* z = ctx.Output<framework::Tensor>("Out");

    using C = platform::complex<T>;
    z->mutable_data<C>(ctx.GetPlace());

// NOTE(chenfeiyu): be careful of the caveats of calling elementwise-related
// facility functions
#if defined(__NVCC__) || defined(__HIPCC__)
    ElementwiseComputeEx<RealAndImagToComplexFunctor<T>, DeviceContext, T, C>(
        ctx, x, y, /*axis*/ -1, RealAndImagToComplexFunctor<T>(), z);
#else
    auto x_dims = x->dims();
    auto y_dims = y->dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseComputeEx<RealAndImagToComplexFunctor<T>, DeviceContext, T, C>(
          ctx, x, y, /*axis*/ -1, RealAndImagToComplexFunctor<T>(), z);
    } else {
      ElementwiseComputeEx<ImagAndRealToComplexFunctor<T>, DeviceContext, T, C>(
          ctx, x, y, /*axis*/ -1, ImagAndRealToComplexFunctor<T>(), z);
    }
#endif
  }
};

template <typename DeviceContext, typename T>
class ComplexGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    using C = platform::complex<T>;

    // skip out in a hacky way
    auto* out = dout;
    ElemwiseGradCompute<DeviceContext, T, ComplexGradForRealFunctor<T>,
                        ComplexGradForImagFunctor<T>, C>(
        ctx, *x, *y, *out, *dout, /*axis*/ -1, dx, dy,
        ComplexGradForRealFunctor<T>(), ComplexGradForImagFunctor<T>());
  }
};

}  // namespace operators
}  // namespace paddle
