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
#ifdef PADDLE_WITH_XPU

#include <memory>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* predict = ctx.Input<phi::DenseTensor>("Predicted");
    auto* labels = ctx.Input<phi::DenseTensor>("Labels");
    auto* loss = ctx.Output<phi::DenseTensor>("Loss");
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));
    loss->mutable_data<T>(ctx.GetPlace());
    int n = predict->numel();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::log_loss(dev_ctx.x_context(),
                          predict->data<T>(),
                          labels->data<T>(),
                          loss->data<T>(),
                          n,
                          epsilon);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "log_loss");
  }
};
template <typename DeviceContext, typename T, typename AttrType = T>
class LogLossGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* predict = ctx.Input<phi::DenseTensor>("Predicted");
    auto* labels = ctx.Input<phi::DenseTensor>("Labels");
    auto* dloss = ctx.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    auto* dpred =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Predicted"));
    if (!dpred) {
      return;
    }
    auto epsilon = static_cast<T>(ctx.Attr<AttrType>("epsilon"));
    dpred->mutable_data<T>(ctx.GetPlace());
    int n = predict->numel();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::log_loss_grad(dev_ctx.x_context(),
                               predict->data<T>(),
                               labels->data<T>(),
                               dloss->data<T>(),
                               dpred->data<T>(),
                               n,
                               epsilon);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "log_loss_grad");
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    log_loss, ops::LogLossXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    log_loss_grad,
    ops::LogLossGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
