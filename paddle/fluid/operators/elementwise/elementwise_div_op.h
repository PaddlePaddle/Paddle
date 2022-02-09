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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void default_elementwise_sub(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          SubFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseSubFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseSubFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T>
void default_elementwise_div(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          DivFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseDivFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseDivFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T>
class ElementwiseDivKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    int axis = ctx.Attr<int>("axis");
    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
    auto pt_z = paddle::experimental::MakePtenDenseTensor(*z);
    pten::DivideRawKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *pt_x.get(), *pt_y.get(), axis, pt_z.get());
  }
};

template <typename T>
struct DivGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout / y; }
};

template <typename T>
struct DivGradDX<paddle::platform::complex<T>> {
  HOSTDEVICE paddle::platform::complex<T> operator()(
      paddle::platform::complex<T> x, paddle::platform::complex<T> y,
      paddle::platform::complex<T> out,
      paddle::platform::complex<T> dout) const {
    paddle::platform::complex<T> y_conj(y.real, -y.imag);
    return dout / y_conj;
  }
};

template <typename T>
struct DivGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return -dout * out / y;
  }
};

template <typename T>
struct DivGradDY<paddle::platform::complex<T>> {
  HOSTDEVICE paddle::platform::complex<T> operator()(
      paddle::platform::complex<T> x, paddle::platform::complex<T> y,
      paddle::platform::complex<T> out,
      paddle::platform::complex<T> dout) const {
    paddle::platform::complex<T> out_div_y_conj((out / y).real,
                                                -(out / y).imag);
    return -dout * out_div_y_conj;
  }
};

template <typename T>
struct DivDoubleDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return y * out * dout - x * dout;
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
ElementwiseDivGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");

  ElemwiseGradCompute<DeviceContext, T, DivGradDX<T>, DivGradDY<T>>(
      ctx, *x, *y, *out, *dout, axis, dx, dy, DivGradDX<T>(), DivGradDY<T>());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
ElementwiseDivGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy);
#endif

template <typename DeviceContext, typename T>
class ElementwiseDivGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    ElementwiseDivGrad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
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
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Out");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()), tensor.place(),
          tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivDoubleGradKernel : public framework::OpKernel<T> {
  using Tensor = framework::Tensor;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Input<Tensor>("Out");
    auto* ddX = ctx.Input<Tensor>("DDX");
    auto* ddY = ctx.Input<Tensor>("DDY");
    auto* dX = ctx.Input<Tensor>("DX");

    auto* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* dOut = ctx.Output<Tensor>("DOut");
    auto* ddOut = ctx.Output<Tensor>("DDOut");

    int axis = ctx.Attr<int>("axis");

    if (dY) dY->mutable_data<T>(Y->dims(), ctx.GetPlace());
    if (dOut) dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());

    // ddX_safe == null ? 0 : ddX
    // ddY_safe == null ? 0 : ddY
    Tensor ddX_safe, ddY_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, dX, ddX, &ddX_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, Y, ddY, &ddY_safe);

    // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
    // dY = Out * dX * ddY / Y - dX * ddX / Y
    // dOut = - dX * ddY
    // To save memory, (1) dout can be used as 'tmp' tensor, (2) ddout can
    // inplace ddx
    Tensor tmp;
    if (dOut) {
      tmp = *dOut;
    } else {
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      tmp = ctx.AllocateTmpTensor<T, DeviceContext>(Out->dims(), dev_ctx);
    }
    if (dY) {
      // dX_div_Y = dX / Y;
      Tensor dX_div_Y = tmp;
      default_elementwise_div<DeviceContext, T>(ctx, dX, Y, &dX_div_Y);

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
      // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
      default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_safe, &tmp);
      default_elementwise_sub<DeviceContext, T>(ctx, &ddX_safe, &tmp, &tmp);
      default_elementwise_div<DeviceContext, T>(ctx, &tmp, Y, ddOut);
    }

    if (dOut) {
      // dOut = - dX * ddY
      default_elementwise_mul<DeviceContext, T>(ctx, dX, &ddY_safe, dOut);
      auto& place =
          *ctx.template device_context<DeviceContext>().eigen_device();
      auto dout = framework::EigenVector<T>::Flatten(*dOut);
      dout.device(place) = static_cast<T>(-1) * dout;
    }
  }
};

}  // namespace operators
}  // namespace paddle
