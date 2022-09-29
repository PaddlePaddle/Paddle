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
      x_var->IsType<framework::LoDTensor>(),
      true,
      platform::errors::InvalidArgument(
          "XPU only support LoDTensor, Input(X) is not LoDTensor"));

  auto x = x_var->Get<framework::LoDTensor>();
  auto* y = ctx.Input<framework::LoDTensor>("Y");
  auto* z = ctx.Output<framework::LoDTensor>("Out");
  int axis = ctx.Attr<int>("axis");

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();
  phi::XPUElementwise<T, XPUType>(dev_ctx, x, *y, axis, z, func);
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
  auto* x = ctx.Input<phi::DenseTensor>("X");
  auto* y = ctx.Input<phi::DenseTensor>("Y");
  auto* dz = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
  auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
  auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
  int axis = ctx.Attr<int>("axis");

  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();
  phi::XPUElementwiseGrad<T, XPUType>(
      dev_ctx, *x, *y, *dz, axis, dx, dy, func, use_x_y_data);
}

}  // namespace operators
}  // namespace paddle
#endif
