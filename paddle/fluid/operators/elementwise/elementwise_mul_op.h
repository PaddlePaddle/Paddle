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
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename DeviceContext, typename T>
void default_elementwise_mul(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                        MulFunctor<T>(), z);
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_mul(const framework::ExecutionContext& ctx,
                const framework::Tensor* x, const framework::Tensor* y,
                framework::Tensor* z) {
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  blas.VMUL(x->numel(), x->data<T>(), y->data<T>(),
            z->mutable_data<T>(ctx.GetPlace()));
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value ||
    !std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_mul(const framework::ExecutionContext& ctx,
                const framework::Tensor* x, const framework::Tensor* y,
                framework::Tensor* z) {
  default_elementwise_mul<DeviceContext, T>(ctx, x, y, z);
}

template <typename DeviceContext, typename T>
class ElementwiseMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE(x_var != nullptr,
                   "Cannot get input Variable X, variable name = %s",
                   ctx.op().Input("X"));
    auto* y = ctx.Input<framework::LoDTensor>("Y");

    framework::Tensor x, *z;
    if (x_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE(y->dims().size() == 1 && y->dims()[0] == 1,
                     "For elementwise_op, if X is Sparse, Y must be scalar.");
      auto& x_sele = x_var->Get<framework::SelectedRows>();
      auto out_sele = ctx.Output<framework::SelectedRows>("Out");
      x = x_sele.value();
      out_sele->set_rows(x_sele.rows());
      out_sele->set_height(x_sele.height());
      out_sele->mutable_value()->Resize(x_sele.value().dims());
      out_sele->mutable_value()->mutable_data(ctx.GetPlace(), x.type());
      z = ctx.Output<framework::SelectedRows>("Out")->mutable_value();
    } else if (x_var->IsType<framework::LoDTensor>()) {
      x = x_var->Get<framework::LoDTensor>();
      z = ctx.Output<framework::LoDTensor>("Out");
    } else {
      PADDLE_THROW("X's type[%s] is not supported by elementwise_op.",
                   framework::ToTypeName(x_var->Type()));
    }

    z->mutable_data<T>(ctx.GetPlace());
    if (x.numel() == y->numel()) {
      elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
    } else {
      default_elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
    }
  }
};

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
  // HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return
  // static_cast<T>(0); }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
  // HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return
  // static_cast<T>(0); }
};

template <typename DeviceContext, typename T>
class ElementwiseMulGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = dout;  // out is not necessary
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, MulGradDX<T>(), MulGradDY<T>());
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>("DOut");
    auto* ddx = ctx.Input<Tensor>("DDX");
    auto* ddy = ctx.Input<Tensor>("DDY");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* ddout = ctx.Output<Tensor>("DDOut");

    dx->mutable_data<T>(x->dims(), ctx.GetPlace());
    dy->mutable_data<T>(x->dims(), ctx.GetPlace());
    ddout->mutable_data<T>(x->dims(), ctx.GetPlace());

    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(ctx.template device_context<DeviceContext>(), dx,
             static_cast<T>(0));
    set_zero(ctx.template device_context<DeviceContext>(), dy,
             static_cast<T>(0));
    set_zero(ctx.template device_context<DeviceContext>(), ddout,
             static_cast<T>(0));

    auto x_t = framework::EigenVector<T>::Flatten(*x);
    auto y_t = framework::EigenVector<T>::Flatten(*y);
    auto dout_t = framework::EigenVector<T>::Flatten(*dout);
    auto dx_t = framework::EigenVector<T>::Flatten(*dx);
    auto dy_t = framework::EigenVector<T>::Flatten(*dy);
    auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    if (ddx) {
      auto ddx_t = framework::EigenVector<T>::Flatten(*ddx);
      ddout_t.device(place) = ddout_t + ddx_t * y_t;
      dy_t.device(place) = dout_t * ddx_t - dout_t * x_t;
    }
    if (ddy) {
      auto ddy_t = framework::EigenVector<T>::Flatten(*ddy);
      ddout_t.device(place) = ddout_t + ddy_t * x_t;
      dx_t.device(place) = dout_t * ddy_t - dout_t * y_t;
    } else {
      dx_t.device(place) = dout_t * y_t * static_cast<T>(-1);
    }
  }
};

}  // namespace operators
}  // namespace paddle
