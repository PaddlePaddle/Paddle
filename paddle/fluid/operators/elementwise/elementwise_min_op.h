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

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
namespace paddle {
namespace operators {

template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a < b ? a : b; }
};

template <typename DeviceContext, typename T>
class ElementwiseMinKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<MinFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MinFunctor<T>(), z);
  }
};

template <typename T>
struct MinGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * (x < y);
  }
};

template <typename T>
struct MinGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * (x >= y);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMinGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* out = dout;  // Fake out, not used
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, MinGradDx<T>, MinGradDy<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, MinGradDx<T>(), MinGradDy<T>());
  }
};
}  // namespace operators
}  // namespace paddle
