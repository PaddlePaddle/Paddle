/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/kernels/xpu/elementwise.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

template <typename T, typename XPUType>
void XPUElementwise(const framework::ExecutionContext& ctx,
                    std::function<int(xpu::Context*,
                                      const XPUType*,
                                      const XPUType*,
                                      XPUType*,
                                      const std::vector<int>&,
                                      const std::vector<int>&)> func) {
  auto x_var = ctx.InputVar("X");
  PADDLE_ENFORCE_NE(
      x_var,
      nullptr,
      platform::errors::InvalidArgument("Cannot get input Variable X"));
  PADDLE_ENFORCE_EQ(
<<<<<<< HEAD
      x_var->IsType<framework::LoDTensor>(),
=======
      x_var->IsType<phi::DenseTensor>(),
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
      true,
      platform::errors::InvalidArgument(
          "XPU only support LoDTensor, Input(X) is not LoDTensor"));

  auto x = x_var->Get<phi::DenseTensor>();
  auto* y = ctx.Input<phi::DenseTensor>("Y");
  auto* z = ctx.Output<phi::DenseTensor>("Out");
  int axis = ctx.Attr<int>("axis");
<<<<<<< HEAD
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  PADDLE_ENFORCE_GE(
      axis,
      0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));
  std::vector<int> x_dims_vec(max_dim, 1);
  std::vector<int> y_dims_vec(max_dim, 1);
  if (x_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      x_dims_vec[i] = x_dims[i];
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      x_dims_vec[i + axis] = x_dims[i];
    }
  }
  if (y_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      y_dims_vec[i] = y_dims[i];
    }
  } else {
    for (int i = 0; i < y_dims.size(); i++) {
      y_dims_vec[i + axis] = y_dims[i];
    }
  }
  const T* x_data = x.data<T>();
  const T* y_data = y->data<T>();
  T* z_data = z->data<T>();

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  int ret = xpu::SUCCESS;

  ret = func(dev_ctx.x_context(),
             reinterpret_cast<const XPUType*>(x_data),
             reinterpret_cast<const XPUType*>(y_data),
             reinterpret_cast<XPUType*>(z_data),
             x_dims_vec,
             y_dims_vec);
  PADDLE_ENFORCE_EQ(
      ret,
      xpu::SUCCESS,
      platform::errors::External(
          "XPU kernel Elementwise occur error in XPUElementwise error code ",
          ret,
          XPUAPIErrorMsg[ret]));
=======

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();
  phi::XPUElementwise<T, XPUType>(dev_ctx, x, *y, axis, z, func);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}

template <typename T, typename XPUType>
void XPUElementwiseGrad(const framework::ExecutionContext& ctx,
                        std::function<int(xpu::Context*,
                                          const XPUType*,
                                          const XPUType*,
                                          const XPUType*,
                                          const XPUType*,
                                          XPUType*,
                                          XPUType*,
                                          const std::vector<int>&,
                                          const std::vector<int>&)> func,
                        bool use_x_y_data) {
<<<<<<< HEAD
  auto* x = ctx.Input<framework::Tensor>("X");
  auto* y = ctx.Input<framework::Tensor>("Y");
  auto* dz = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
  auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
  auto* z = dz;
  int axis = ctx.Attr<int>("axis");
  const framework::DDim& x_dims = x->dims();
  const framework::DDim& y_dims = y->dims();
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));
  std::vector<int> x_dims_vec(max_dim, 1);
  std::vector<int> y_dims_vec(max_dim, 1);
  if (x_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      x_dims_vec[i] = x_dims[i];
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      x_dims_vec[i + axis] = x_dims[i];
    }
  }
  if (y_dims.size() == max_dim) {
    for (int i = 0; i < max_dim; i++) {
      y_dims_vec[i] = y_dims[i];
    }
  } else {
    for (int i = 0; i < y_dims.size(); i++) {
      y_dims_vec[i + axis] = y_dims[i];
    }
  }

  const T* x_data = use_x_y_data ? x->data<T>() : z->data<T>();
  const T* y_data = use_x_y_data ? y->data<T>() : z->data<T>();
  const T* z_data = z->data<T>();
=======
  auto* x = ctx.Input<phi::DenseTensor>("X");
  auto* y = ctx.Input<phi::DenseTensor>("Y");
  auto* dz = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
  auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
  auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
  int axis = ctx.Attr<int>("axis");
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();
<<<<<<< HEAD

  if (dx) {
    dx_data = dx->mutable_data<T>(ctx.GetPlace());
  }
  if (dy) {
    dy_data = dy->mutable_data<T>(ctx.GetPlace());
  }

  int ret = func(dev_ctx.x_context(),
                 reinterpret_cast<const XPUType*>(x_data),
                 reinterpret_cast<const XPUType*>(y_data),
                 reinterpret_cast<const XPUType*>(z_data),
                 reinterpret_cast<const XPUType*>(dz_data),
                 reinterpret_cast<XPUType*>(dy_data),
                 reinterpret_cast<XPUType*>(dx_data),
                 x_dims_vec,
                 y_dims_vec);
  PADDLE_ENFORCE_EQ(
      ret,
      xpu::SUCCESS,
      platform::errors::External(
          "XPU kernel Elementwise occur error in XPUElementwise error code ",
          ret,
          XPUAPIErrorMsg[ret]));
=======
  phi::XPUElementwiseGrad<T, XPUType>(
      dev_ctx, *x, *y, *dz, axis, dx, dy, func, use_x_y_data);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
}

}  // namespace operators
}  // namespace paddle
#endif
