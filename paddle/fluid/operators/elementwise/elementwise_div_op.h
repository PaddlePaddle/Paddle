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

#include <vector>
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

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

template <typename T>
struct DivDoubleDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return y * out * dout - x * dout;
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

    // ddX_safe == null ? 0 : ddX
    // ddY_safe == null ? 0 : ddY
    Tensor ddX_safe, ddY_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, Out, ddX, &ddX_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, Y, ddY, &ddY_safe);

    // TODO(dengkaipeng): here we cached median calculation results
    // in 3 tensors to improve running speed, if memory optimize 
    // is required, you could try to deleted these tesnors.
    
    // ddX_div_Y = ddX / Y
    Tensor ddX_div_Y;
    ddX_div_Y.mutable_data<T>(Out->dims(), ctx.GetPlace());
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
        ctx, &ddX_safe, Y, axis, DivFunctor<T>(), &ddX_div_Y);

    // ddY_div_Y = ddY / Y
    Tensor ddY_div_Y;
    ddY_div_Y.mutable_data<T>(Y->dims(), ctx.GetPlace());
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
        ctx, &ddY_safe, Y, 0, DivFunctor<T>(), &ddY_div_Y);

    // dX_div_Y = dX / Y;
    Tensor dX_div_Y;
    dX_div_Y.mutable_data<T>(Out->dims(), ctx.GetPlace());
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
        ctx, dX, Y, axis, DivFunctor<T>(), &dX_div_Y);

    if (dOut) {
      // dOut = - dX * ddY
      default_elementwise_mul<DeviceContext, T>(ctx, dX, &ddY_safe, dOut);
      auto dout = framework::EigenVector<T>::Flatten(*dOut);
      dout.device(place) = static_cast<T>(-1) * dout;
    }

    if (dY) {
      // NOTE(dengkaipeng): in the following ElemwiseGradCompute, for the
      // first output tensor is nullptr, the branch to calculate first
      // output tensor will not be activated, DivGradDx function will not
      // be called and can be ignored, the first branch has little effect
      // on running speed.

      // dY = Out * dX * ddY / Y - dX * ddX / Y
      ElemwiseGradCompute<DeviceContext, T, DivGradDX<T>, DivDoubleDY<T>>(
          ctx, ddX_safe, ddY_safe, *Out, dX_div_Y, axis, nullptr, dY,
          DivGradDX<T>(), DivDoubleDY<T>());
    }

    if (ddOut) {
      // ddOut = ddX / Y - Out * ddY / Y
      default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_div_Y, ddOut);
      auto ddx_div_y = framework::EigenVector<T>::Flatten(ddX_div_Y);
      auto ddout = framework::EigenVector<T>::Flatten(*ddOut);
      ddout.device(place) = static_cast<T>(-1) * ddout + ddx_div_y;
    }
  }
};

}  // namespace operators
}  // namespace paddle
