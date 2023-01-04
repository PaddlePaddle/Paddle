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
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
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
                       const paddle::optional<DenseTensor>& ddx,
                       const paddle::optional<DenseTensor>& ddy,
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

    dev_ctx.template Alloc<T>(ddout);
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
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
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

    dev_ctx.template Alloc<T>(ddout);
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
                            const paddle::optional<DenseTensor>& ddx,
                            const paddle::optional<DenseTensor>& ddy,
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
template <typename T, typename Context>
void ElementwiseFMaxGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out_grad,
                               int axis,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  funcs::ElementwiseGradPreProcess(out_grad, x_grad);

  auto out = out_grad;  // Fake out, not used
  auto x_dim = x.dims();
  auto y_dim = y.dims();
  if (x.dims() == y.dims()) {
    funcs::ElemwiseGradComputeNoBroadcast<Context,
                                          T,
                                          funcs::FMaxGradDx<T>,
                                          funcs::FMaxGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMaxGradDx<T>(),
        funcs::FMaxGradDy<T>());
  } else {
    funcs::ElemwiseGradComputeWithBroadcast<T,
                                            funcs::FMaxGradDx<T>,
                                            funcs::FMaxGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMaxGradDx<T>(),
        funcs::FMaxGradDy<T>());
  }
}

template <typename T, typename Context>
void ElementwiseFMinGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               const DenseTensor& out_grad,
                               int axis,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  funcs::ElementwiseGradPreProcess(out_grad, x_grad);
  auto out = out_grad;  // Fake out, not used
  auto x_dim = x.dims();
  auto y_dim = y.dims();
  if (x.dims() == y.dims()) {
    funcs::ElemwiseGradComputeNoBroadcast<Context,
                                          T,
                                          funcs::FMinGradDx<T>,
                                          funcs::FMinGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMinGradDx<T>(),
        funcs::FMinGradDy<T>());
  } else {
    funcs::ElemwiseGradComputeWithBroadcast<T,
                                            funcs::FMinGradDx<T>,
                                            funcs::FMinGradDy<T>>(
        dev_ctx,
        x_dim,
        y_dim,
        x,
        y,
        out,
        out_grad,
        axis,
        x_grad,
        y_grad,
        funcs::FMinGradDx<T>(),
        funcs::FMinGradDy<T>());
  }
}

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

// avoid [-Wint-in-bool-context] warning
template <>
struct MulGradDX<bool> {
  HOSTDEVICE bool operator()(bool x, bool y, bool out, bool dout) const {
    return dout && y;
  }
};

template <typename T>
struct MulGradDX<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> y_conj(y.real, -y.imag);
    return dout * y_conj;
  }
};

/*
******************************
    Multiply Grad
******************************
*/

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
};

// avoid [-Wint-in-bool-context] warning
template <>
struct MulGradDY<bool> {
  HOSTDEVICE bool operator()(bool x, bool y, bool out, bool dout) const {
    return dout && x;
  }
};

template <typename T>
struct MulGradDY<phi::dtype::complex<T>> {
  HOSTDEVICE phi::dtype::complex<T> operator()(
      phi::dtype::complex<T> x,
      phi::dtype::complex<T> y,
      phi::dtype::complex<T> out,
      phi::dtype::complex<T> dout) const {
    phi::dtype::complex<T> x_conj(x.real, -x.imag);
    return dout * x_conj;
  }
};

template <typename T, typename Context>
void MultiplyDoubleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy,
                              DenseTensor* ddout) {
  if (ddout) dev_ctx.template Alloc<T>(ddout);

  DenseTensor ddx_safe, ddy_safe;
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, x, ddx.get_ptr(), &ddx_safe);
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, y, ddy.get_ptr(), &ddy_safe);

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
    auto& place = *dev_ctx.eigen_device();
    // size(ddout) > size(ddx) or we don't have ddx, ddout can't use memory of
    // ddx using inplace

    bool without_ddx = (ddx.get_ptr() == nullptr);
    if (!without_ddx) {
      without_ddx = (ddout->numel() > ddx.get_ptr()->numel());
    }
    if (without_ddx) {
      phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
          dev_ctx,
          ddx_safe,
          ddy_safe,
          dout,
          dout,
          axis,
          dx,
          dy,
          MulGradDX<T>(),
          MulGradDY<T>());

      DenseTensor ddout_tmp;
      ddout_tmp.Resize(ddout->dims());
      dev_ctx.template Alloc<T>(&ddout_tmp);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, y, ddx_safe, ddout, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, x, &ddout_tmp, axis);

      auto ddout_t = phi::EigenVector<T>::Flatten(*ddout);
      auto ddout_tmp_t = phi::EigenVector<T>::Flatten(ddout_tmp);
      ddout_t.device(place) = ddout_t + ddout_tmp_t;
    } else {
      // use dx to save memory, other than alloc tmp tensor
      if (dx) {
        DenseTensor* ddout_tmp = dx;
        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, x, ddy_safe, ddout_tmp, axis);

        // NOTE: in the following ElemwiseGradCompute, for the
        // first output tensor is nullptr, the branch to calculate first
        // output tensor will not be activated, DivGradDx function will not
        // be called and can be ignored, the first branch has little effect
        // on running speed.
        phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
            dev_ctx,
            ddx_safe,
            ddy_safe,
            dout,
            dout,
            axis,
            nullptr,
            dy,
            MulGradDX<T>(),
            MulGradDY<T>());

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, ddx_safe, y, ddout, axis);

        auto ddout_t = phi::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = phi::EigenVector<T>::Flatten(*ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, dout, ddy_safe, dx, axis);

      } else {
        DenseTensor tmp_a(ddout->dtype());
        tmp_a.Resize(ddout->dims());

        dev_ctx.template Alloc<T>(&tmp_a);

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, x, ddy_safe, &tmp_a, axis);

        auto ddout_t1 = phi::EigenVector<T>::Flatten(tmp_a);

        funcs::DefaultElementwiseOperator<Context,
                                          T,
                                          funcs::MultiplyFunctor<T>,
                                          funcs::InverseMultiplyFunctor<T>>(
            dev_ctx, ddx_safe, y, ddout, axis);

        auto ddout_t2 = phi::EigenVector<T>::Flatten(*ddout);
        ddout_t2.device(place) = ddout_t2 + ddout_t1;
      }
    }
  } else {
    if (dx && dy) {
      phi::funcs::ElemwiseGradCompute<Context, T, MulGradDX<T>, MulGradDY<T>>(
          dev_ctx,
          ddx_safe,
          ddy_safe,
          dout,
          dout,
          axis,
          dx,
          dy,
          MulGradDX<T>(),
          MulGradDY<T>());
    }
  }
}

template <typename T, typename Context>
void MultiplyTripleGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              const paddle::optional<DenseTensor>& ddx,
                              const paddle::optional<DenseTensor>& ddy,
                              const paddle::optional<DenseTensor>& d_dx,
                              const paddle::optional<DenseTensor>& d_dy,
                              const paddle::optional<DenseTensor>& d_ddout,
                              int axis,
                              DenseTensor* d_x,
                              DenseTensor* d_y,
                              DenseTensor* d_dout,
                              DenseTensor* d_ddx,
                              DenseTensor* d_ddy) {
  if (d_x) {
    d_x->Resize(x.dims());
    dev_ctx.template Alloc<T>(d_x);
  }
  if (d_y) {
    d_y->Resize(y.dims());
    dev_ctx.template Alloc<T>(d_y);
  }
  if (d_dout) {
    d_dout->Resize(dout.dims());
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_ddx) {
    d_ddx->Resize(x.dims());
    dev_ctx.template Alloc<T>(d_ddx);
  }
  if (d_ddy) {
    d_ddy->Resize(y.dims());
    dev_ctx.template Alloc<T>(d_ddy);
  }

  auto& place = *dev_ctx.eigen_device();

  DenseTensor ddx_safe, ddy_safe;
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, x, ddx.get_ptr(), &ddx_safe);
  funcs::GetDoubleGradSafeTensor<Context, T>(
      dev_ctx, y, ddy.get_ptr(), &ddy_safe);

  if (d_ddout.get_ptr()) {
    if (d_x) {
      // d_x = ddy * d_ddout
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, *(d_ddout.get_ptr()), d_x, axis);
    }
    if (d_y) {
      // d_y = ddx * d_ddout
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddx_safe, *(d_ddout.get_ptr()), d_y, axis);
    }
  } else {
    if (d_x) {
      FullLikeKernel<T, Context>(dev_ctx, x, Scalar(0.0), x.dtype(), d_x);
    }
    if (d_y) {
      FullLikeKernel<T, Context>(dev_ctx, y, Scalar(0.0), y.dtype(), d_y);
    }
  }

  if (d_dout) {
    // get d_dout
    // d_dout = ddy * d_dx + d_dy * ddx
    DenseTensor d_dout_tmp;
    d_dout_tmp.Resize(dout.dims());
    dev_ctx.template Alloc<T>(&d_dout_tmp);

    if (d_dy && d_dx) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, d_dy.get(), ddx_safe, d_dout, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, d_dx.get(), &d_dout_tmp, axis);

      auto d_dout_t = phi::EigenVector<T>::Flatten(*d_dout);
      auto d_dout_tmp_t = phi::EigenVector<T>::Flatten(d_dout_tmp);
      d_dout_t.device(place) = d_dout_t + d_dout_tmp_t;
    } else if (d_dy && !d_dx) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, d_dy.get(), ddx_safe, d_dout, axis);
      auto d_dout_t = phi::EigenVector<T>::Flatten(*d_dout);
      d_dout_t.device(place) = d_dout_t;
    } else if (!d_dy && d_dx) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, ddy_safe, d_dx.get(), d_dout, axis);

      auto d_dout_t = phi::EigenVector<T>::Flatten(*d_dout);
      d_dout_t.device(place) = d_dout_t;
    } else {
      FullLikeKernel<T, Context>(
          dev_ctx, dout, Scalar(0.0), dout.dtype(), d_dout);
    }
  }

  if (d_ddx && ddx) {
    // get d_ddx
    // d_ddx = dout * d_dy + y * d_ddout
    DenseTensor d_ddx_tmp;
    d_ddx_tmp.Resize(ddx->dims());
    dev_ctx.template Alloc<T>(&d_ddx_tmp);
    if (d_dy && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dy.get(), d_ddx, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, y, *(d_ddout.get_ptr()), &d_ddx_tmp, axis);

      auto d_ddx_t = phi::EigenVector<T>::Flatten(*d_ddx);
      auto d_ddx_tmp_t = phi::EigenVector<T>::Flatten(d_ddx_tmp);
      d_ddx_t.device(place) = d_ddx_t + d_ddx_tmp_t;
    } else if (d_dy && !d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dy.get(), d_ddx, axis);

      auto d_ddx_t = phi::EigenVector<T>::Flatten(*d_ddx);
      d_ddx_t.device(place) = d_ddx_t;
    } else if (!d_dy && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, y, *(d_ddout.get_ptr()), d_ddx, axis);

      auto d_ddx_t = phi::EigenVector<T>::Flatten(*d_ddx);
      d_ddx_t.device(place) = d_ddx_t;
    } else {
      FullLikeKernel<T, Context>(dev_ctx, x, Scalar(0.0), x.dtype(), d_ddx);
    }
  }

  if (d_ddy && ddy) {
    // get d_ddy
    // d_ddy = dout * d_dx + x * d_ddout
    DenseTensor d_ddy_tmp;
    d_ddy_tmp.Resize(ddy->dims());
    dev_ctx.template Alloc<T>(&d_ddy_tmp);

    if (d_dx && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dx.get(), d_ddy, axis);

      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, x, *(d_ddout.get_ptr()), &d_ddy_tmp, axis);

      auto d_ddy_t = phi::EigenVector<T>::Flatten(*d_ddy);
      auto d_ddy_tmp_t = phi::EigenVector<T>::Flatten(d_ddy_tmp);
      d_ddy_t.device(place) = d_ddy_t + d_ddy_tmp_t;
    } else if (d_dx && !d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, dout, d_dx.get(), d_ddy, axis);

      auto d_ddy_t = phi::EigenVector<T>::Flatten(*d_ddy);
      d_ddy_t.device(place) = d_ddy_t;
    } else if (!d_dx && d_ddout) {
      funcs::DefaultElementwiseOperator<Context,
                                        T,
                                        funcs::MultiplyFunctor<T>,
                                        funcs::InverseMultiplyFunctor<T>>(
          dev_ctx, x, *(d_ddout.get_ptr()), d_ddy, axis);

      auto d_ddy_t = phi::EigenVector<T>::Flatten(*d_ddy);
      d_ddy_t.device(place) = d_ddy_t;
    } else {
      FullLikeKernel<T, Context>(dev_ctx, y, Scalar(0.0), y.dtype(), d_ddy);
    }
  }
}

/*
******************************
    Maximum Grad
******************************
*/

template <typename T>
struct MaxGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x > y);
  }
};

template <typename T>
struct MaxGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x <= y);
  }
};

/*
******************************
    Minimum Grad
******************************
*/
template <typename T>
struct MinGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x < y);
  }
};

template <typename T>
struct MinGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x >= y);
  }
};

template <typename T>
struct HeavisideGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(0);
  }
};

template <typename T>
struct HeavisideGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x == static_cast<T>(0));
  }
};

template <typename T, typename Context>
void ElementwiseHeavisideGradKernel(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& y,
                                    const DenseTensor& dout,
                                    int axis,
                                    DenseTensor* dx,
                                    DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  phi::funcs::
      ElemwiseGradCompute<Context, T, HeavisideGradDx<T>, HeavisideGradDy<T>>(
          dev_ctx,
          x,
          y,
          dout,
          dout,
          axis,
          dx,
          dy,
          HeavisideGradDx<T>(),
          HeavisideGradDy<T>());
}

template <typename T>
struct PowGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    if (std::is_integral<T>::value) {
      return dout * y *
             std::pow(static_cast<double>(x), static_cast<double>(y - 1));
    }
#endif
    return dout * y * std::pow(x, y - 1);
  }
};

template <>
struct PowGradDX<dtype::float16> {
  HOSTDEVICE dtype::float16 operator()(dtype::float16 x,
                                       dtype::float16 y,
                                       dtype::float16 out,
                                       dtype::float16 dout) const {
    float tmp_y = static_cast<float>(y);
    float tmp_dout = static_cast<float>(dout);
    float tmp_x = static_cast<float>(x);
    float result = tmp_dout * tmp_y * std::pow(tmp_x, tmp_y - 1.0f);
    return static_cast<dtype::float16>(result);
  }
};

template <typename T, typename Enable = void>
struct PowGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
    if (std::is_integral<T>::value) {
      return dout * std::log(static_cast<double>(x)) *
             std::pow(static_cast<double>(x), static_cast<double>(y));
    }
#endif
    return dout * std::log(x) * std::pow(x, y);
  }
};

template <>
struct PowGradDY<dtype::float16, void> {
  HOSTDEVICE dtype::float16 operator()(dtype::float16 x,
                                       dtype::float16 y,
                                       dtype::float16 out,
                                       dtype::float16 dout) const {
    float tmp_y = static_cast<float>(y);
    float tmp_dout = static_cast<float>(dout);
    float tmp_x = static_cast<float>(x);
    float tmp_pow = std::pow(tmp_x, tmp_y);
    float result = tmp_pow * tmp_dout * std::log(tmp_x);
    return static_cast<dtype::float16>(result);
  }
};

template <typename T, typename Context>
void ElementwisePowGradKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy) {
  funcs::ElementwiseGradPreProcess(dout, dx);
  phi::funcs::ElemwiseGradCompute<Context, T, PowGradDX<T>, PowGradDY<T>>(
      dev_ctx, x, y, dout, dout, axis, dx, dy, PowGradDX<T>(), PowGradDY<T>());
}

}  // namespace phi
