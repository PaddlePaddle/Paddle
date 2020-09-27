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

#include "paddle/fluid/operators/math/softmax.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/softmax_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class SoftmaxXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    Out->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, X->dims());
    const int d = SizeFromAxis(axis, X->dims());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::softmax2d_forward(dev_ctx.x_context(), X->data<float>(),
                                   Out->data<float>(), n, d, d <= 2048);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::Fatal("XPU kernel softmax2d_forward run failed!"));
  }
};

template <typename DeviceContext, typename T>
class SoftmaxGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    const int rank = dX->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    // allocate memory on device.
    dX->mutable_data<T>(context.GetPlace());

    const int n = SizeToAxis(axis, dX->dims());
    const int d = SizeFromAxis(axis, dX->dims());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r =
        xpu::softmax2d_backward(dev_ctx.x_context(), Out->data<float>(),
                                dOut->data<float>(), dX->data<float>(), n, d);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::Fatal("XPU kernel softmax2d_backward run failed!"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    softmax, ops::SoftmaxXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    softmax_grad,
    ops::SoftmaxGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
