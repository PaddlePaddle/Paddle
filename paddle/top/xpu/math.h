/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_XPU

#include "paddle/top/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/xpu/xpu_header.h"

namespace pt {

using XPUContext = paddle::platform::XPUDeviceContext;

template <typename T>
void Sign(const XPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  T* out_data = out->mutable_data<T>();
  auto xpu_ctx = dev_ctx.x_context();
  int r = xpu::activation_forward(
      xpu_ctx, xpu::Activation_t::SIGN, x.numel(), x.data<T>(), out_data);
  PADDLE_ENFORCE_EQ(r,
                    xpu::Error_t::SUCCESS,
                    paddle::platform::errors::Fatal("XPU sign kernel error!"));
}

template <typename T>
void Mean(const XPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  T* out_data = out->mutable_data<T>();
  auto xpu_ctx = dev_ctx.x_context();
  const T* x_data = x.data<T>();
  int r = xpu::mean(xpu_ctx, x_data, out_data, x.numel());
  PADDLE_ENFORCE_EQ(
      r,
      xpu::Error_t::SUCCESS,
      paddle::platform::errors::External(
          "XPU kernel error, Mean op execution not succeed, error code=%d", r));
}

template <typename T>
void Scale(const XPUContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  T* out_data = out->mutable_data<T>();
  PADDLE_ENFORCE_EQ(x.dims(),
                    out->dims(),
                    paddle::platform::errors::InvalidArgument(
                        "In and out should have the same dim,"
                        " expected %s, but got %s.",
                        x.dims().to_str().c_str(),
                        out->dims().to_str().c_str()));
  int r = xpu::scale(dev_ctx.x_context(),
                     x.data<T>(),
                     out_data,
                     x.numel(),
                     bias_after_scale,
                     scale,
                     bias);
  PADDLE_ENFORCE_EQ(
      r,
      XPU_SUCCESS,
      paddle::platform::errors::External(
          "XPU scale kernel return wrong value[%d %s]", r, XPUAPIErrorMsg[r]));
}

}  // namespace pt

#endif
