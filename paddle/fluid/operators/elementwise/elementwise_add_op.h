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

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

#include "paddle/pten/kernels/math_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
void LaunchBroadcastElementwiseCpuKernel(const framework::ExecutionContext &ctx,
                                         const framework::Tensor *x,
                                         const framework::Tensor *y,
                                         framework::Tensor *z) {
  int axis = ctx.Attr<int>("axis");
  auto x_dims = x->dims();
  auto y_dims = y->dims();
  if (x_dims.size() >= y_dims.size()) {
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          AddFunctor<T>(), z);
  } else {
    ElementwiseComputeEx<InverseAddFunctor<T>, DeviceContext, T>(
        ctx, x, y, axis, InverseAddFunctor<T>(), z);
  }
}

template <typename DeviceContext, typename T, class Enable = void>
struct SameDimsElemwiseAdd {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z);
};

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.device_context<DeviceContext>();
    int axis = ctx.Attr<int>("axis");
    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
    auto pt_z = paddle::experimental::MakePtenDenseTensor(*z);
    pten::AddRawKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE &>(dev_ctx),
        *pt_x.get(), *pt_y.get(), axis, pt_z.get());
  }
};

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
default_elementwise_add_grad(const framework::ExecutionContext &ctx,
                             const framework::Tensor *x,
                             const framework::Tensor *y,
                             const framework::Tensor *out,
                             const framework::Tensor *dout,
                             framework::Tensor *dx, framework::Tensor *dy) {
  int axis = ctx.Attr<int>("axis");

  ElemwiseExplicitGradCompute<DeviceContext, T, IdentityGrad<T>,
                              IdentityGrad<T>>(ctx, *x, *y, *out, *dout, axis,
                                               dx, dy, IdentityGrad<T>(),
                                               IdentityGrad<T>());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     const framework::Tensor *out,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy) {
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  if (dx) {
    blas.VCOPY(dout->numel(), dout->data<T>(),
               dx->mutable_data<T>(ctx.GetPlace()));
  }

  if (dy) {
    blas.VCOPY(dout->numel(), dout->data<T>(),
               dy->mutable_data<T>(ctx.GetPlace()));
  }
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     const framework::Tensor *out,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy) {
  default_elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// cuda definition
template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
elementwise_add_grad(const framework::ExecutionContext &ctx,
                     const framework::Tensor *x, const framework::Tensor *y,
                     const framework::Tensor *out,
                     const framework::Tensor *dout, framework::Tensor *dx,
                     framework::Tensor *dy);

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_same<DeviceContext, platform::CUDADeviceContext>::value>::type
default_elementwise_add_grad(const framework::ExecutionContext &ctx,
                             const framework::Tensor *x,
                             const framework::Tensor *y,
                             const framework::Tensor *out,
                             const framework::Tensor *dout,
                             framework::Tensor *dx, framework::Tensor *dy);
#endif

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);

    using Tensor = framework::Tensor;

    auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Input<Tensor>("Y");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    // skip out
    auto *out = dout;

    // Special case when dy is not needed and dx doesn't reduce
    if (dx != nullptr && dy == nullptr && dx->dims() == dout->dims()) {
      VLOG(4) << "Special case when dy is not needed and dx doesn't "
                 "reduce";
      framework::TensorCopy(
          *dout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), dx);
    } else if (dx == nullptr && dy != nullptr && dy->dims() == dout->dims()) {
      VLOG(4) << "Special case when dx is not needed and dy doesn't "
                 "reduce";
      framework::TensorCopy(
          *dout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), dy);
    } else if (dx != nullptr && dy != nullptr && (dx->dims() == dy->dims())) {
      elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx, dy);
    } else {
      default_elementwise_add_grad<DeviceContext, T>(ctx, x, y, out, dout, dx,
                                                     dy);
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseAddDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using Tensor = framework::Tensor;

    auto *y = ctx.Input<Tensor>("Y");
    auto *dout = ctx.Input<Tensor>("DOut");
    auto *ddx = ctx.Input<Tensor>("DDX");
    auto *ddy = ctx.Input<Tensor>("DDY");

    auto *ddout = ctx.Output<Tensor>("DDOut");

    // ddOut = ddx + ddy
    if (ddout) {
      Tensor ddx_safe, ddy_safe;
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, dout, ddx, &ddx_safe);
      GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);

      ddout->mutable_data<T>(ctx.GetPlace());
      LaunchBroadcastElementwiseCpuKernel<DeviceContext, T>(ctx, &ddx_safe,
                                                            &ddy_safe, ddout);
    }
  }
};

template <typename DeviceContext, typename T>
class ElementwiseAddTripleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using Tensor = framework::Tensor;
    auto *ddx = ctx.Input<Tensor>("DDX");
    auto *ddy = ctx.Input<Tensor>("DDY");
    auto *d_ddout = ctx.Input<Tensor>("D_DDOut");
    auto *d_ddx = ctx.Output<Tensor>("D_DDX");
    auto *d_ddy = ctx.Output<Tensor>("D_DDY");
    // skip out
    auto *out = d_ddout;

    // Special case when d_ddy is not needed and d_ddx doesn't reduce
    if (d_ddx != nullptr && d_ddy == nullptr &&
        d_ddx->dims() == d_ddout->dims()) {
      VLOG(4) << "Special case when d_ddy is not needed and d_ddx doesn't "
                 "reduce";
      framework::TensorCopy(
          *d_ddout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), d_ddx);
    } else if (d_ddx == nullptr && d_ddy != nullptr &&
               d_ddy->dims() == d_ddout->dims()) {
      VLOG(4) << "Special case when d_ddx is not needed and d_ddy doesn't "
                 "reduce";
      framework::TensorCopy(
          *d_ddout, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), d_ddy);
    } else if (d_ddx != nullptr && d_ddy != nullptr &&
               (d_ddx->dims() == d_ddy->dims())) {
      elementwise_add_grad<DeviceContext, T>(ctx, ddx, ddy, out, d_ddout, d_ddx,
                                             d_ddy);
    } else {
      default_elementwise_add_grad<DeviceContext, T>(ctx, ddx, ddy, out,
                                                     d_ddout, d_ddx, d_ddy);
    }
  }
};

}  // namespace operators
}  // namespace paddle
