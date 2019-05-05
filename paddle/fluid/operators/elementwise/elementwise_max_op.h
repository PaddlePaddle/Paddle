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
struct MaxFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename DeviceContext, typename T>
class ElementwiseMaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");

    PADDLE_ENFORCE_NOT_NULL(x);
    PADDLE_ENFORCE_NOT_NULL(y);
    PADDLE_ENFORCE_NOT_NULL(z);

    // Compute the output's dims and share x's LoD
    z->mutable_data<T>(x->dims(), ctx.GetPlace());
    z->set_lod(x->lod());

    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<MaxFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MaxFunctor<T>(), z);
  }
};

template <typename T>
struct MaxGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * (x > y);
  }
};

template <typename T>
struct MaxGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * (x <= y);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMaxGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);

    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* dout = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
    auto* out = dout;  // Fake out, not used
    int axis = ctx.Attr<int>("axis");

    PADDLE_ENFORCE_NOT_NULL(dout);

    if (dx) {
      // Compute the dx's dims and share dout's LoD
      dx->mutable_data<T>(dout->dims(), ctx.GetPlace());
      dx->set_lod(dout->lod());
    }
    if (dy) {
      // Compute the dy's dims and share y's LoD
      dy->mutable_data<T>(y->dims(), ctx.GetPlace());
      dy->set_lod(y->lod());
    }

    ElemwiseGradCompute<DeviceContext, T, MaxGradDx<T>, MaxGradDy<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, MaxGradDx<T>(), MaxGradDy<T>());
  }
};
}  // namespace operators
}  // namespace paddle
