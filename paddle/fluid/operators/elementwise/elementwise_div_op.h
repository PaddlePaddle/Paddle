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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
namespace paddle {
namespace operators {

template <typename T>
struct DivFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a / b; }
};

template <typename DeviceContext, typename T>
class ElementwiseDivKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          DivFunctor<T>(), z);
  }
};

template <typename T>
struct DivGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout / y; }
};

template <typename T>
struct DivGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return -dout * out / y;
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");

    auto* x = dout;  // Fake x, not used

    ElemwiseGradCompute<DeviceContext, T, DivGradDX<T>, DivGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, DivGradDX<T>(), DivGradDY<T>());
  }
};

class ElementwiseDivOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput("DOut")) {
      ctx->ShareDim("DX", "DOut");
      ctx->ShareLoD("DX", "DOut");
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", y_grad_name);
      ctx->ShareLoD("Y", y_grad_name);
    }
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DX", "DDOut");
      ctx->ShareLoD("DX", "DDOut");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("DDX")->type();

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Input<Tensor>("Out");
    auto* ddX = ctx.Input<Tensor>("DDX");
    auto* ddY = ctx.Input<Tensor>("DDY");
    auto* dX = ctx.Input<Tensor>("DX");

    auto* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* dOut = ctx.Output<Tensor>("DOut");
    auto* ddOut = ctx.Output<Tensor>("DDOut");

    if (dY) dY->mutable_data<T>(Y->dims(), ctx.GetPlace());
    if (dOut) dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    int axis = ctx.Attr<int>("axis");

    Tensor ddX_tmp, ddY_tmp;
    if (ddX) {
      ddX_tmp.mutable_data<T>(ddX->dims(), ctx.GetPlace());
      ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
          ctx, ddX, Y, axis, DivFunctor<T>(), &ddX_tmp);
    }
    if (ddY) {
      ddY_tmp.mutable_data<T>(ddY->dims(), ctx.GetPlace());
      ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
          ctx, ddY, Y, axis, DivFunctor<T>(), &ddY_tmp);
    }
    if (dOut) {
      if (ddY) {
        Tensor dOut_tmp;
        dOut_tmp.mutable_data<T>(dOut->dims(), ctx.GetPlace());

        default_elementwise_mul<DeviceContext, T>(ctx, dX, ddY, &dOut_tmp);
        auto dout = framework::EigenVector<T>::Flatten(*dOut);
        auto dout_tmp = framework::EigenVector<T>::Flatten(dOut_tmp);
        dout.device(place) = static_cast<T>(-1) * dout_tmp;
      }
    }
    if (dY) {
      auto dy = framework::EigenVector<T>::Flatten(*dY);
      if (ddX && ddY) {
        Tensor dY_tmp1, dY_tmp2, tmp;

        dY_tmp1.mutable_data<T>(dY->dims(), ctx.GetPlace());
        dY_tmp2.mutable_data<T>(dY->dims(), ctx.GetPlace());
        tmp.mutable_data<T>(Out->dims(), ctx.GetPlace());

        default_elementwise_mul<DeviceContext, T>(ctx, &ddX_tmp, dX, &dY_tmp1);
        default_elementwise_mul<DeviceContext, T>(ctx, dX, &ddY_tmp, &tmp);
        default_elementwise_mul<DeviceContext, T>(ctx, Out, &tmp, &dY_tmp2);
        auto dy_tmp1 = framework::EigenVector<T>::Flatten(dY_tmp1);
        auto dy_tmp2 = framework::EigenVector<T>::Flatten(dY_tmp2);
        dy.device(place) = static_cast<T>(-1) * dy_tmp1 + dy_tmp2;
      } else {
        if (ddX) {
          Tensor dY_tmp3;
          dY_tmp3.mutable_data<T>(Y->dims(), ctx.GetPlace());

          default_elementwise_mul<DeviceContext, T>(ctx, &ddX_tmp, dX, &dY_tmp3);
          auto dy_tmp1 = framework::EigenVector<T>::Flatten(dY_tmp3);
          dy.device(place) = static_cast<T>(-1) * dy_tmp1;
        }
        if (ddY) {
          Tensor dY_tmp2, tmp;
          dY_tmp2.mutable_data<T>(dY->dims(), ctx.GetPlace());
          tmp.mutable_data<T>(dY->dims(), ctx.GetPlace());

          default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_tmp, &tmp);
          default_elementwise_mul<DeviceContext, T>(ctx, dX, &tmp, &dY_tmp2);
          auto dy_tmp2 = framework::EigenVector<T>::Flatten(dY_tmp2);
          dy.device(place) = dy_tmp2;
        }
      }
    }
    if (ddOut) {
      if (ddX && ddY) {
        Tensor ddOut_tmp;
        ddOut_tmp.mutable_data<T>(dOut->dims(), ctx.GetPlace());

        default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_tmp, &ddOut_tmp);
        auto ddout_tmp2 = framework::EigenVector<T>::Flatten(ddX_tmp);
        auto ddout_tmp = framework::EigenVector<T>::Flatten(ddOut_tmp);
        auto ddout = framework::EigenVector<T>::Flatten(*ddOut);
        ddout.device(place) = ddout_tmp + static_cast<T>(-1) * ddout_tmp2;
      } else {
        if (ddX) {
          ddOut = &ddX_tmp;
        }
        if (ddY) {
          Tensor ddOut_tmp;
          ddOut_tmp.mutable_data<T>(dOut->dims(), ctx.GetPlace());
          default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_tmp,
                                                    &ddOut_tmp);
          auto ddout_tmp = framework::EigenVector<T>::Flatten(ddOut_tmp);
          auto ddout = framework::EigenVector<T>::Flatten(*ddOut);
          ddout.device(place) = static_cast<T>(-1) * ddout_tmp;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
