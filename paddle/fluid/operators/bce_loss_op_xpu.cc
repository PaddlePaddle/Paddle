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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class XPUBCELossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* labels = context.Input<Tensor>("Label");
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto x_numel = x->numel();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::bce_loss<T>(dev_ctx.x_context(), x->data<T>(),
                             labels->data<T>(), out->data<T>(), x_numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "bce_loss");
  }
};

template <typename DeviceContext, typename T>
class XPUBCELossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* labels = context.Input<Tensor>("Label");
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(context.GetPlace());

    auto x_numel = x->numel();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::bce_loss_grad<T>(dev_ctx.x_context(), x->data<T>(),
                                  labels->data<T>(), dout->data<T>(),
                                  dx->data<T>(), x_numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "bce_loss_grad");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    bce_loss, ops::XPUBCELossKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    bce_loss_grad,
    ops::XPUBCELossGradKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
