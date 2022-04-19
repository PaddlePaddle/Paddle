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

#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GeluXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    const XPUType* x_data = reinterpret_cast<const XPUType*>(x->data<T>());
    XPUType* y_data = reinterpret_cast<XPUType*>(out->mutable_data<T>(place));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::gelu<XPUType>(dev_ctx.x_context(), x_data, y_data, x->numel());
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU gelu kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class GeluGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();
    const XPUType* x_data = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* dout_data =
        reinterpret_cast<const XPUType*>(dout->data<T>());
    XPUType* dx_data = reinterpret_cast<XPUType*>(dx->mutable_data<T>(place));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int r = xpu::gelu_grad<XPUType>(dev_ctx.x_context(), x_data, nullptr,
                                    dout_data, dx_data, dout->numel());
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU gelu_grad kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    gelu, ops::GeluXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::GeluXPUKernel<paddle::platform::XPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_XPU_KERNEL(
    gelu_grad,
    ops::GeluGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::GeluGradXPUKernel<paddle::platform::XPUDeviceContext,
                           paddle::platform::float16>);
