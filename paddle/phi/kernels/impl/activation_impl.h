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
#include "paddle/phi/kernels/funcs/blas/blas.h"

#include "paddle/fluid/platform/device_context.h"

namespace phi {

#define ToString(x) #x

template <typename T, typename Context, typename Functor>
void ActivationImpl(const Context& dev_ctx,
                    const DenseTensor& X,
                    DenseTensor* Out,
                    const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(Out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(Out);
  auto x = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&X, "Input", "X", "Activation"));
  auto out = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(Out, "Output", "Out", "Activation"));
  auto* place = dev_ctx.eigen_device();
  // use 32bit index to speed up computation
  bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = paddle::platform::is_gpu_place(dev_ctx.GetPlace());
  if (use_32bit_index && is_gpu_place) {
    functor(*place, To32BitIndex(x), To32BitIndex(out));
  } else {
    functor(*place, x, out);
  }
}

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

}  // namespace phi
