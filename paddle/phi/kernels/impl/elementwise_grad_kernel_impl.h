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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
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

}  // namespace phi
