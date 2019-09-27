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
elementwise_mul_same_dims(const framework::ExecutionContext& ctx,
                          const framework::Tensor* x,
                          const framework::Tensor* y, framework::Tensor* z) {
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  blas.VMUL(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value ||
    !std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_mul_same_dims(const framework::ExecutionContext& ctx,
                          const framework::Tensor* x,
                          const framework::Tensor* y, framework::Tensor* z) {
  auto eigen_x = framework::EigenVector<T>::Flatten(*x);
  auto eigen_y = framework::EigenVector<T>::Flatten(*y);
  auto eigen_z = framework::EigenVector<T>::Flatten(*z);

  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  eigen_z.device(place) = eigen_x * eigen_y;
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
      elementwise_mul_same_dims<DeviceContext, T>(ctx, &x, y, z);
    } else {
      default_elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
    }
  }
};

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
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

    if (ddout) ddout->mutable_data<T>(ctx.GetPlace());

    Tensor ddx_safe, ddy_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, x, ddx, &ddx_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

    // dx = dout * ddy
    // dy = dout * ddx
    // ddout = ddx * y + x * ddy
    // change computation sequence to save memory, so ddout can inplace ddx and
    // dx can be used as 'tmp' tensor
    // (1) dx = x * ddy
    // (2) dy = dout * ddx
    // (3) ddout = ddx * y
    // (4) ddout = ddout + dx
    // (5) dx = dout *ddy
    if (ddout) {
      // use dx to save memory, other than alloc tmp tensor
      Tensor* ddout_tmp = dx;

      default_elementwise_mul<DeviceContext, T>(ctx, x, &ddy_safe, ddout_tmp);
      int axis = ctx.Attr<int>("axis");
      // NOTE: in the following ElemwiseGradCompute, for the
      // first output tensor is nullptr, the branch to calculate first
      // output tensor will not be activated, DivGradDx function will not
      // be called and can be ignored, the first branch has little effect
      // on running speed.
      ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
          ctx, ddx_safe, ddy_safe, *dout, *dout, axis, nullptr, dy,
          MulGradDX<T>(), MulGradDY<T>());
      default_elementwise_mul<DeviceContext, T>(ctx, &ddx_safe, y, ddout);

      auto& place =
          *ctx.template device_context<DeviceContext>().eigen_device();
      auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);
      auto ddout_tmp_t = framework::EigenVector<T>::Flatten(*ddout_tmp);
      ddout_t.device(place) = ddout_t + ddout_tmp_t;
      default_elementwise_mul<DeviceContext, T>(ctx, dout, &ddy_safe, dx);
    }
  }
};

DECLARE_INPLACE_OP_INFERER(ElementwiseMulDoubleGradOpInplace, {"DDX", "DDOut"});

}  // namespace operators
}  // namespace paddle
