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

#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/cpu_info.h"

#include "paddle/pten/kernels/math_kernel.h"

namespace paddle {
namespace operators {

class ElementwiseMulOp : public ElementwiseOp {
 public:
  using Tensor = framework::Tensor;
  using ElementwiseOp::ElementwiseOp;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

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
void default_elementwise_mul(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          MulFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseMulFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseMulFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseMul {
  void operator()(const framework::ExecutionContext& ctx,
                  const framework::Tensor* x, const framework::Tensor* y,
                  framework::Tensor* z);
};

template <typename DeviceContext, typename T>
class ElementwiseMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE_EQ(x_var != nullptr, true,
                      platform::errors::InvalidArgument(
                          "Cannot get input Variable X, Variable name = %s.",
                          ctx.InputName("X")));
    auto* y = ctx.Input<framework::LoDTensor>("Y");

    framework::Tensor x, *z;
    if (x_var->IsType<pten::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(y->dims().size() == 1 && y->dims()[0] == 1, true,
                        platform::errors::InvalidArgument(
                            "For elementwise_op, if X is Sparse, Y must be "
                            "scalar. But reveived the size of Y = %s.",
                            y->dims().size()));
      auto& x_sele = x_var->Get<pten::SelectedRows>();
      auto out_sele = ctx.Output<pten::SelectedRows>("Out");
      x = x_sele.value();
      out_sele->set_rows(x_sele.rows());
      out_sele->set_height(x_sele.height());
      out_sele->mutable_value()->Resize(x_sele.value().dims());
      out_sele->mutable_value()->mutable_data(ctx.GetPlace(), x.type());
      z = ctx.Output<pten::SelectedRows>("Out")->mutable_value();
      z->mutable_data<T>(ctx.GetPlace());
      auto dims_equal = x.dims() == y->dims();
      if (dims_equal) {
        SameDimsElemwiseMul<DeviceContext, T> same_dims_mul;
        same_dims_mul(ctx, &x, y, z);
      } else {
        default_elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
      }
    } else if (x_var->IsType<framework::LoDTensor>()) {
      auto* x_lod = ctx.Input<framework::LoDTensor>("X");
      auto* z_lod = ctx.Output<framework::LoDTensor>("Out");
      z_lod->mutable_data<T>(ctx.GetPlace());

      auto& dev_ctx = ctx.device_context<DeviceContext>();
      int axis = ctx.Attr<int>("axis");
      auto pt_x = paddle::experimental::MakePtenDenseTensor(*x_lod);
      auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
      auto pt_z = paddle::experimental::MakePtenDenseTensor(*z_lod);
      pten::MultiplyRawKernel<T>(
          static_cast<const typename framework::ConvertToPtenContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *pt_x.get(), *pt_y.get(), axis, pt_z.get());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "X's type[%s] is not supported by elementwise_op. X's type should be "
          "LoDTensor or SelectedRows.",
          framework::ToTypeName(x_var->Type())));
    }
  }
};
template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

template <typename T>
struct MulGradDX<paddle::platform::complex<T>> {
  HOSTDEVICE paddle::platform::complex<T> operator()(
      paddle::platform::complex<T> x, paddle::platform::complex<T> y,
      paddle::platform::complex<T> out,
      paddle::platform::complex<T> dout) const {
    paddle::platform::complex<T> y_conj(y.real, -y.imag);
    return dout * y_conj;
  }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
};

template <typename T>
struct MulGradDY<paddle::platform::complex<T>> {
  HOSTDEVICE paddle::platform::complex<T> operator()(
      paddle::platform::complex<T> x, paddle::platform::complex<T> y,
      paddle::platform::complex<T> out,
      paddle::platform::complex<T> dout) const {
    paddle::platform::complex<T> x_conj(x.real, -x.imag);
    return dout * x_conj;
  }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
ElementwiseMulGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy) {
  int axis = ctx.Attr<int>("axis");
  ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
      ctx, *x, *y, *out, *dout, axis, dx, dy, MulGradDX<T>(), MulGradDY<T>());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
ElementwiseMulGrad(const framework::ExecutionContext& ctx,
                   const framework::Tensor* x, const framework::Tensor* y,
                   const framework::Tensor* out, const framework::Tensor* dout,
                   framework::Tensor* dx, framework::Tensor* dy);
#endif

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

    ElementwiseMulGrad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
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
    // (5) dx = dout * ddy
    if (ddout) {
      int axis = ctx.Attr<int>("axis");
      auto& place =
          *ctx.template device_context<DeviceContext>().eigen_device();
      // size(ddout) > size(ddx), ddout can't use memory of ddx using inplace
      if (ddout->numel() > ddx->numel()) {
        ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
            ctx, ddx_safe, ddy_safe, *dout, *dout, axis, dx, dy, MulGradDX<T>(),
            MulGradDY<T>());

        Tensor ddout_tmp;
        ddout_tmp.mutable_data<T>(ddout->dims(), ctx.GetPlace());

        default_elementwise_mul<DeviceContext, T>(ctx, y, &ddx_safe, ddout);
        default_elementwise_mul<DeviceContext, T>(ctx, &ddy_safe, x,
                                                  &ddout_tmp);

        auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = framework::EigenVector<T>::Flatten(ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;
      } else {
        // use dx to save memory, other than alloc tmp tensor
        Tensor* ddout_tmp = dx;

        default_elementwise_mul<DeviceContext, T>(ctx, x, &ddy_safe, ddout_tmp);
        // NOTE: in the following ElemwiseGradCompute, for the
        // first output tensor is nullptr, the branch to calculate first
        // output tensor will not be activated, DivGradDx function will not
        // be called and can be ignored, the first branch has little effect
        // on running speed.
        ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
            ctx, ddx_safe, ddy_safe, *dout, *dout, axis, nullptr, dy,
            MulGradDX<T>(), MulGradDY<T>());
        default_elementwise_mul<DeviceContext, T>(ctx, &ddx_safe, y, ddout);

        auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = framework::EigenVector<T>::Flatten(*ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;
        default_elementwise_mul<DeviceContext, T>(ctx, dout, &ddy_safe, dx);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulTripleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;
    // get input
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>("DOut");
    auto* ddx = ctx.Input<framework::Tensor>("DDX");
    auto* ddy = ctx.Input<framework::Tensor>("DDY");

    auto* d_dx = ctx.Input<framework::Tensor>("D_DX");
    auto* d_dy = ctx.Input<framework::Tensor>("D_DY");
    auto* d_ddout = ctx.Input<framework::Tensor>("D_DDOut");

    // get output
    auto* out_d_x = ctx.Output<framework::Tensor>("D_X");
    auto* out_d_y = ctx.Output<framework::Tensor>("D_Y");
    auto* out_d_dout = ctx.Output<framework::Tensor>("D_DOut");

    auto* out_d_ddx = ctx.Output<framework::Tensor>("D_DDX");
    auto* out_d_ddy = ctx.Output<framework::Tensor>("D_DDY");

    if (out_d_x) out_d_x->mutable_data<T>(x->dims(), ctx.GetPlace());
    if (out_d_y) out_d_y->mutable_data<T>(y->dims(), ctx.GetPlace());
    if (out_d_dout) out_d_dout->mutable_data<T>(dout->dims(), ctx.GetPlace());
    if (out_d_ddx) out_d_ddx->mutable_data<T>(x->dims(), ctx.GetPlace());
    if (out_d_ddy) out_d_ddy->mutable_data<T>(y->dims(), ctx.GetPlace());

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    Tensor ddx_safe, ddy_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, x, ddx, &ddx_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

    if (d_ddout) {
      if (out_d_x) {
        // out_d_x = ddy * d_ddout
        default_elementwise_mul<DeviceContext, T>(ctx, &ddy_safe, d_ddout,
                                                  out_d_x);
      }
      if (out_d_y) {
        // out_d_y = ddx * d_ddout
        default_elementwise_mul<DeviceContext, T>(ctx, &ddx_safe, d_ddout,
                                                  out_d_y);
      }
    }

    if (out_d_dout) {
      // get out_d_dout
      // out_d_dout = ddy * d_dx + d_dy * ddx
      Tensor out_d_dout_tmp;
      out_d_dout_tmp.mutable_data<T>(dout->dims(), ctx.GetPlace());
      default_elementwise_mul<DeviceContext, T>(ctx, d_dy, &ddx_safe,
                                                out_d_dout);
      default_elementwise_mul<DeviceContext, T>(ctx, &ddy_safe, d_dx,
                                                &out_d_dout_tmp);
      auto out_d_dout_t = framework::EigenVector<T>::Flatten(*out_d_dout);
      auto out_d_dout_tmp_t =
          framework::EigenVector<T>::Flatten(out_d_dout_tmp);
      out_d_dout_t.device(place) = out_d_dout_t + out_d_dout_tmp_t;
    }

    if (out_d_ddx) {
      // get out_d_ddx
      // out_d_ddx = dout * d_dy + y * d_ddout
      Tensor out_d_ddx_tmp;
      out_d_ddx_tmp.mutable_data<T>(ddx->dims(), ctx.GetPlace());
      default_elementwise_mul<DeviceContext, T>(ctx, dout, d_dy, out_d_ddx);
      default_elementwise_mul<DeviceContext, T>(ctx, y, d_ddout,
                                                &out_d_ddx_tmp);
      auto out_d_ddx_t = framework::EigenVector<T>::Flatten(*out_d_ddx);
      auto out_d_ddx_tmp_t = framework::EigenVector<T>::Flatten(out_d_ddx_tmp);
      out_d_ddx_t.device(place) = out_d_ddx_t + out_d_ddx_tmp_t;
    }

    if (out_d_ddy) {
      // get out_d_ddy
      // out_d_ddy = dout * d_dx + x * d_ddout
      Tensor out_d_ddy_tmp;
      out_d_ddy_tmp.mutable_data<T>(ddy->dims(), ctx.GetPlace());
      default_elementwise_mul<DeviceContext, T>(ctx, dout, d_dx, out_d_ddy);
      default_elementwise_mul<DeviceContext, T>(ctx, x, d_ddout,
                                                &out_d_ddy_tmp);
      auto out_d_ddy_t = framework::EigenVector<T>::Flatten(*out_d_ddy);
      auto out_d_ddy_tmp_t = framework::EigenVector<T>::Flatten(out_d_ddy_tmp);
      out_d_ddy_t.device(place) = out_d_ddy_t + out_d_ddy_tmp_t;
    }
  }
};
}  // namespace operators
}  // namespace paddle
