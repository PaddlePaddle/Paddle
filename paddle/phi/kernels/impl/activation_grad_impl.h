// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

#include "paddle/fluid/platform/device_context.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGradImpl(const Context& dev_ctx,
                        const DenseTensor* X,
                        const DenseTensor* Out,
                        const DenseTensor* dOut,
                        DenseTensor* dX,
                        const Functor& functor) {
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    PADDLE_ENFORCE_NOT_NULL(
        Out, errors::NotFound("The input DenseTensor Out can not be nullptr"));
  }
  PADDLE_ENFORCE_NOT_NULL(
      dOut, errors::NotFound("The input DenseTensor dOut can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      dX, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (!Out) {
    Out = dOut;  // fake out
  }
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    PADDLE_ENFORCE_NOT_NULL(
        X, errors::NotFound("The input DenseTensor X can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    X = dX;
  }

  dev_ctx.template Alloc<T>(dX);
  auto dout = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(dOut, "Input", "Out@GRAD", "ActivationGrad"));
  auto out = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(Out, "Input", "Out", "ActivationGrad"));
  auto dx = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(dX, "Input", "X@GRAD", "ActivationGrad"));
  auto x = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(X, "Input", "X", "ActivationGrad"));
  auto* place = dev_ctx.eigen_device();
  // use 32bit index to speed up computation
  bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = paddle::platform::is_gpu_place(dev_ctx.GetPlace());
  if (use_32bit_index && is_gpu_place) {
    functor(*place,
            To32BitIndex(x),
            To32BitIndex(out),
            To32BitIndex(dout),
            To32BitIndex(dx));
  } else {
    functor(*place, x, out, dout, dx);
  }
}

template <typename T, typename Context, typename Functor>
void ActivationDoubleGradImpl(const Context& dev_ctx,
                              const DenseTensor* X,
                              const DenseTensor* Out,
                              const DenseTensor* ddX,
                              DenseTensor* dX,
                              DenseTensor* dOut,
                              DenseTensor* ddOut,
                              const Functor& functor) {
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    PADDLE_ENFORCE_NOT_NULL(
        X, errors::NotFound("The input DenseTensor X can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    X = ddX;
  }
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    PADDLE_ENFORCE_NOT_NULL(
        Out, errors::NotFound("The input DenseTensor Out can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    Out = ddX;
  }

  if (ddOut) {
    dev_ctx.template Alloc<T>(ddOut);
  }
  if (dOut) {
    dev_ctx.template Alloc<T>(dOut);
  }
  if (dX) {
    dX->Resize(Out->dims());
    dev_ctx.template Alloc<T>(dX);
  }

  functor(dev_ctx, X, Out, ddX, ddOut, dOut, dX);
}

template <typename T, typename Context>
void ReluDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          DenseTensor* ddout) {
  funcs::ReluGradGradFunctor<T> relu_double_grad_functor;
  ActivationDoubleGradImpl<T, Context, funcs::ReluGradGradFunctor<T>>(
      dev_ctx,
      nullptr,
      &out,
      &ddx,
      nullptr,
      nullptr,
      ddout,
      relu_double_grad_functor);
}

template <typename T, typename Context>
void LeakyReluDoubleGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& ddx,
                               float alpha,
                               DenseTensor* ddout) {
  funcs::LeakyReluGradGradFunctor<T> leaky_relu_double_grad_functor;
  leaky_relu_double_grad_functor.alpha = alpha;
  ActivationDoubleGradImpl<T, Context, funcs::LeakyReluGradGradFunctor<T>>(
      dev_ctx,
      &x,
      nullptr,
      &ddx,
      nullptr,
      nullptr,
      ddout,
      leaky_relu_double_grad_functor);
}

template <typename T, typename Context>
void TanhDoubleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          const DenseTensor& dout,
                          DenseTensor* dout_new,
                          DenseTensor* ddout) {
  if (dout_new) {
    dout_new->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout_new);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::TanhGradGradFunctor<T> functor;
  functor(dev_ctx, &out, &ddx, &dout, dout_new, ddout);
}

template <typename T, typename Context>
void TanhTripleGradKernel(const Context& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& ddx,
                          const DenseTensor& dout,
                          const DenseTensor& d_ddout,
                          const DenseTensor& d_dout_new,
                          DenseTensor* d_out_new,
                          DenseTensor* d_dout,
                          DenseTensor* d_ddx) {
  if (d_dout) {
    d_dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_out_new) {
    d_dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_out_new);
  }
  if (d_ddx) {
    d_dout->Resize(ddx.dims());
    dev_ctx.template Alloc<T>(d_ddx);
  }
  funcs::TanhTripleGradFunctor<T> functor;
  functor(dev_ctx,
          &out,
          &ddx,
          &dout,
          &d_ddout,
          &d_dout_new,  // input
          d_dout,
          d_out_new,
          d_ddx);  // output
}

template <typename T, typename Context>
void EluDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         float alpha,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::ELUGradGradFunctor<T> functor;
  functor.alpha = alpha;
  functor(dev_ctx, &x, &ddx, ddout, &dout, dx);
}

template <typename T, typename Context>
void LogitGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out_grad,
                     float eps,
                     DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  auto eigen_x = EigenVector<T>::Flatten(x);
  auto eigen_dout = EigenVector<T>::Flatten(out_grad);
  auto eigen_dx = EigenVector<T>::Flatten(*x_grad);
  auto& place = *dev_ctx.eigen_device();
  auto eigen_p = EigenVector<T>::Flatten(x);

  funcs::LogitGradFunctor<T> functor;
  functor(place, eigen_x, eigen_dout, eigen_dx, eigen_p, eps);
}

template <typename T, typename Context>
void SigmoidDoubleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out,
                             const DenseTensor& ddx,
                             const DenseTensor& dout,
                             DenseTensor* dout_new,
                             DenseTensor* ddout) {
  if (dout_new) {
    dout_new->Resize(out.dims());
    dev_ctx.template Alloc<T>(dout_new);
  }
  if (ddout) {
    ddout->Resize(out.dims());
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::SigmoidGradGradFunctor<T> functor;
  functor(dev_ctx, &out, &ddx, &dout, dout_new, ddout);
}

template <typename T, typename Context>
void SigmoidTripleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out,
                             const DenseTensor& ddx,
                             const DenseTensor& dout,
                             const DenseTensor& d_ddout,
                             const DenseTensor& d_dout_new,
                             DenseTensor* d_out_new,
                             DenseTensor* d_dout,
                             DenseTensor* d_ddx) {
  if (d_dout) {
    d_dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_dout);
  }
  if (d_out_new) {
    d_dout->Resize(out.dims());
    dev_ctx.template Alloc<T>(d_out_new);
  }
  if (d_ddx) {
    d_dout->Resize(ddx.dims());
    dev_ctx.template Alloc<T>(d_ddx);
  }
  funcs::SigmoidTripleGradFunctor<T> functor;
  functor(dev_ctx,
          &out,
          &ddx,
          &dout,
          &d_ddout,
          &d_dout_new,
          d_dout,
          d_out_new,
          d_ddx);
}

template <typename T, typename Context>
void LogDoubleGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         const DenseTensor& ddx,
                         DenseTensor* dx,
                         DenseTensor* ddout) {
  if (dx) {
    dx->Resize(x.dims());
    dev_ctx.template Alloc<T>(dx);
  }
  if (ddout) {
    dev_ctx.template Alloc<T>(ddout);
  }
  funcs::LogGradGradFunctor<T> functor;
  functor(dev_ctx, &x, &ddx, ddout, &dout, dx);
}

template <typename T, typename Context>
void PowGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& dout,
                   const Scalar& factor,
                   DenseTensor* dx) {
  PADDLE_ENFORCE_NOT_NULL(
      dx, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
  }
  auto dout_flatten = EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&dout, "Input", "Out@GRAD", "PowGrad"));
  auto dx_flatten = EigenVector<T>::Flatten(
      GET_DATA_SAFELY(dx, "Output", "X@GRAD", "PowGrad"));
  auto x_flatten =
      EigenVector<T>::Flatten(GET_DATA_SAFELY(&x, "Input", "X", "PowGrad"));
  auto* place = dev_ctx.eigen_device();
  phi::funcs::PowGradFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = factor.to<float>();
  functor(*place, x_flatten, nullptr, dout_flatten, dx_flatten);
}

}  // namespace phi
