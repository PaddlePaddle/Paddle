/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace phi {

template <typename T, typename Context, typename GradFunc>
void AddGradImpl(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const DenseTensor& out_grad,
                 int axis,
                 DenseTensor* x_grad,
                 DenseTensor* y_grad,
                 GradFunc grad_func) {
  phi::funcs::ElementwiseGradPreProcess(out_grad, x_grad);
  auto* out = &out_grad;
  // Special case when y_grad is not needed and x_grad doesn't reduce
  if (x_grad != nullptr && y_grad == nullptr &&
      x_grad->dims() == out_grad.dims()) {
    VLOG(4) << "Special case when y_grad is not needed and x_grad doesn't "
               "reduce";
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  } else if (x_grad == nullptr && y_grad != nullptr &&
             y_grad->dims() == out_grad.dims()) {
    VLOG(4) << "Special case when x_grad is not needed and y_grad doesn't "
               "reduce";
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, y_grad);
  } else {
    grad_func(dev_ctx, x, y, *out, out_grad, x_grad, y_grad, axis);
  }
}

template <typename T, typename Context>
void AddDoubleGradImpl(const Context& dev_ctx,
                       const DenseTensor& y,
                       const paddle::optional<const DenseTensor&>& ddx,
                       const paddle::optional<const DenseTensor&>& ddy,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* ddout) {
  // ddOut = ddx + ddy
  if (ddout) {
    DenseTensor ddx_safe, ddy_safe;
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, dout, ddx.get_ptr(), &ddx_safe);
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, y, ddy.get_ptr(), &ddy_safe);

    ddout->mutable_data<T>(dev_ctx.GetPlace());
    auto ddx_dims = ddx_safe.dims();
    auto ddy_dims = ddy_safe.dims();
    if (ddx_dims.size() >= ddy_dims.size()) {
      funcs::ElementwiseCompute<funcs::AddFunctor<T>, T>(
          dev_ctx, ddx_safe, ddy_safe, axis, funcs::AddFunctor<T>(), ddout);
    } else {
      funcs::ElementwiseCompute<funcs::InverseAddFunctor<T>, T>(
          dev_ctx,
          ddx_safe,
          ddy_safe,
          axis,
          funcs::InverseAddFunctor<T>(),
          ddout);
    }
  }
}

template <typename T, typename Context>
void SubtractDoubleGradImpl(const Context& dev_ctx,
                            const DenseTensor& y,
                            const paddle::optional<const DenseTensor&>& ddx,
                            const paddle::optional<const DenseTensor&>& ddy,
                            const DenseTensor& dout,
                            int axis,
                            DenseTensor* ddout) {
  // DDOut = ddx - ddy
  if (ddout) {
    DenseTensor ddx_safe, ddy_safe;
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, dout, ddx.get_ptr(), &ddx_safe);
    funcs::GetDoubleGradSafeTensor<Context, T>(
        dev_ctx, y, ddy.get_ptr(), &ddy_safe);

    ddout->mutable_data<T>(dev_ctx.GetPlace());
    funcs::ElementwiseCompute<funcs::SubtractFunctor<T>, T>(
        dev_ctx, ddx_safe, ddy_safe, axis, funcs::SubtractFunctor<T>(), ddout);
  }
}

/*
******************************
    Divide Grad
******************************
*/

template <typename T>
struct DivGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout / y; }
};

template <typename T>
struct DivGradDX<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> y_conj(y.real, -y.imag);
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
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> out_div_y_conj((out / y).real, -(out / y).imag);
    return -dout * out_div_y_conj;
  }
};

template <typename T>
struct DivDoubleDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return y * out * dout - x * dout;
  }
};

template <typename T, typename Context>
void DivideDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& y,
                            const DenseTensor& out,
                            const DenseTensor& dx,
                            paddle::optional<const DenseTensor&> ddx,
                            paddle::optional<const DenseTensor&> ddy,
                            int axis,
                            DenseTensor* dy,
                            DenseTensor* dout,
                            DenseTensor* ddout) {
  if (dy) {
    dy->Resize(y.dims());
    dev_ctx.template Alloc<T>(dy);
  }
  if (dout) {
    dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }
  // ddX_safe == null ? 0 : ddX
  // ddY_safe == null ? 0 : ddY
  DenseTensor ddX_safe, ddY_safe;
  phi::funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, dx, ddx.get_ptr(), &ddX_safe);
  phi::funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, y, ddy.get_ptr(), &ddY_safe);

  // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
  // dY = Out * dX * ddY / Y - dX * ddX / Y
  // dOut = - dX * ddY
  // To save memory, (1) dout can be used as 'tmp' tensor, (2) ddout can
  // inplace ddx
  DenseTensor tmp;
  if (dout) {
    tmp = *dout;
  } else {
    tmp.Resize(out.dims());
    dev_ctx.template Alloc<T>(&tmp);
  }
  if (dy) {
    // dX_div_Y = dX / Y;
    DenseTensor dX_div_Y = tmp;
    funcs::DefaultElementwiseOperator<Context,
                                      T,
                                      funcs::DivideFunctor<T>,
                                      funcs::InverseDivideFunctor<T>>(
        dev_ctx, dx, y, &dX_div_Y, axis);

    // NOTE(dengkaipeng): in the following ElemwiseGradCompute, for the
    // first output tensor is nullptr, the branch to calculate first
    // output tensor will not be activated, DivGradDx function will not
    // be called and can be ignored, the first branch has little effect
    // on running speed.

    // dY = Out * dX * ddY / Y - dX * ddX / Y
    phi::funcs::ElemwiseGradCompute<Context, T, DivGradDX<T>, DivDoubleDY<T>>(
        dev_ctx,
        ddX_safe,
        ddY_safe,
        out,
        dX_div_Y,
        axis,
        nullptr,
        dy,
        DivGradDX<T>(),
        DivDoubleDY<T>());
  }

  if (ddout) {
    // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
    funcs::DefaultElementwiseOperator<Context,
                                      T,
                                      funcs::MultiplyFunctor<T>,
                                      funcs::InverseMultiplyFunctor<T>>(
        dev_ctx, out, ddY_safe, &tmp, axis);
    funcs::DefaultElementwiseOperator<Context,
                                      T,
                                      funcs::SubtractFunctor<T>,
                                      funcs::InverseSubtractFunctor<T>>(
        dev_ctx, ddX_safe, tmp, &tmp, axis);
    funcs::DefaultElementwiseOperator<Context,
                                      T,
                                      funcs::DivideFunctor<T>,
                                      funcs::InverseDivideFunctor<T>>(
        dev_ctx, tmp, y, ddout, axis);
  }

  if (dout) {
    // dOut = - dX * ddY
    funcs::DefaultElementwiseOperator<Context,
                                      T,
                                      funcs::MultiplyFunctor<T>,
                                      funcs::InverseMultiplyFunctor<T>>(
        dev_ctx, dx, ddY_safe, dout, axis);
    auto& place = *dev_ctx.eigen_device();
    auto dout_result = phi::EigenVector<T>::Flatten(*dout);
    dout_result.device(place) = static_cast<T>(-1) * dout_result;
  }
}

}  // namespace phi
