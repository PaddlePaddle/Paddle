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
#include "paddle/fluid/platform/place.h"

#include "paddle/pten/kernels/elementwise_grad_kernel.h"
#include "paddle/pten/kernels/math_kernel.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ElementwiseSubKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    int axis = ctx.Attr<int>("axis");
    pten::SubtractRawKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *x, *y, axis, z);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseSubGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    pten::SubtractGradKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *x, *y, *dout, axis, dx, dy);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseSubDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>("DOut");
    auto* ddx = ctx.Input<Tensor>("DDX");
    auto* ddy = ctx.Input<Tensor>("DDY");

    auto* ddout = ctx.Output<Tensor>("DDOut");
    int axis = ctx.Attr<int>("axis");
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    paddle::optional<const pten::DenseTensor&> ddx_optional = paddle::none;
    paddle::optional<const pten::DenseTensor&> ddy_optional = paddle::none;
    if (ddx != nullptr) {
      ddx_optional = *ddx;
    }
    if (ddy != nullptr) {
      ddy_optional = *ddy;
    }
    pten::SubtractDoubleGradKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *y, ddx_optional, ddy_optional, *dout, axis, ddout);
  }
};

}  // namespace operators
}  // namespace paddle
