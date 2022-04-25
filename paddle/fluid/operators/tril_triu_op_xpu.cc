/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  Licensed under
the Apache License, Version 2.0 (the "License");
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
class TrilTriuXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<framework::Tensor>("X");
    const auto* x_data = x->data<T>();
    auto* out = context.Output<framework::Tensor>("Out");
    auto* out_data = out->mutable_data<T>(context.GetPlace());

    const int diagonal = context.Attr<int>("diagonal");
    const bool lower = context.Attr<bool>("lower");
    auto xshape = phi::vectorize<int>(x->dims());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = 0;
    if (lower) {
      r = xpu::tril(dev_ctx.x_context(), x_data, out_data, xshape, diagonal);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "tril_op");
    } else {
      r = xpu::triu(dev_ctx.x_context(), x_data, out_data, xshape, diagonal);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "triu_op");
    }
  }
};

template <typename DeviceContext, typename T>
class TrilTriuGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const auto* dout_data = d_out->data<T>();
    auto* d_x = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dx_data = d_x->mutable_data<T>(context.GetPlace());

    const int diagonal = context.Attr<int>("diagonal");
    const bool lower = context.Attr<bool>("lower");

    auto dy_shape = phi::vectorize<int>(d_out->dims());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = 0;
    if (lower) {
      r = xpu::tril(dev_ctx.x_context(), dout_data, dx_data, dy_shape,
                    diagonal);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "tril_op");
    } else {
      r = xpu::triu(dev_ctx.x_context(), dout_data, dx_data, dy_shape,
                    diagonal);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "triu_op");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    tril_triu, ops::TrilTriuXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::TrilTriuXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    tril_triu_grad,
    ops::TrilTriuGradXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::TrilTriuGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
