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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/huber_loss_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class HuberLossXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in0 = ctx.Input<Tensor>("X");
    auto* in1 = ctx.Input<Tensor>("Y");
    auto* residual = ctx.Output<Tensor>("Residual");
    auto* out = ctx.Output<Tensor>("Out");
    auto delta = ctx.Attr<float>("delta");

    auto residual_data = residual->mutable_data<T>(ctx.GetPlace());
    auto out_data = out->mutable_data<T>(ctx.GetPlace());
    auto in0_data = in0->data<T>();
    auto in1_data = in1->data<T>();

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int r = xpu::huber_loss<T>(dev_ctx.x_context(), in0_data, in1_data,
                               residual_data, out_data, in0->numel(), 1, delta);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS, platform::errors::External(
                                          "XPU API(huber_loss) return wrong "
                                          "value[%d %s]",
                                          r, XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class HuberLossGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* residual = ctx.Input<Tensor>("Residual");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto delta = ctx.Attr<float>("delta");

    T* dx_data = nullptr;
    T* dy_data = nullptr;
    if (dx) {
      dx_data = dx->mutable_data<T>(ctx.GetPlace());
    }
    if (dy) {
      dy_data = dy->mutable_data<T>(ctx.GetPlace());
    }
    auto dout_data = dout->data<T>();
    auto residual_data = residual->data<T>();
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int r =
        xpu::huber_loss_grad<T>(dev_ctx.x_context(), residual_data, dout_data,
                                dx_data, dy_data, dout->numel(), 1, delta);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU API(huber_loss_grad) return wrong "
                                   "value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(huber_loss, ops::HuberLossXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(huber_loss_grad, ops::HuberLossGradXPUKernel<float>);

#endif
