// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_XPU

#include <memory>
#include <vector>

#include "paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("This kernel only runs on XPU."));

    // input and output data
    auto* input = context.Input<Tensor>("X");
    auto* label = context.Input<Tensor>("Label");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();

    // attrs
    bool normalize = context.Attr<bool>("normalize");
    PADDLE_ENFORCE_EQ(
        normalize, false,
        platform::errors::InvalidArgument("normalize only support true now."));
    int ignore_index = context.Attr<int>("ignore_index");
    PADDLE_ENFORCE_EQ(ignore_index, kIgnoreIndex,
                      platform::errors::InvalidArgument(
                          "ignore_index only support %d now.", kIgnoreIndex));

    int r = xpu::sigmoid_cross_entropy_with_logits(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(input->data<T>()),
        reinterpret_cast<const XPUType*>(label->data<T>()),
        reinterpret_cast<XPUType*>(output->data<T>()), 1, input->numel());
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU sigmoid_cross_entropy_with_logits "
                                   "kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsGradXPUKernel
    : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::Unavailable("This kernel only runs on XPU."));

    // input and output data
    auto* input = context.Input<Tensor>("X");
    auto* label = context.Input<Tensor>("Label");
    auto* dy = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();

    int r = xpu::sigmoid_cross_entropy_with_logits_grad(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(input->data<T>()),
        reinterpret_cast<const XPUType*>(label->data<T>()),
        reinterpret_cast<const XPUType*>(dy->data<T>()),
        reinterpret_cast<XPUType*>(dx->data<T>()), 1, input->numel());
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU sigmoid_cross_entropy_with_logits_grad "
                                   "kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(sigmoid_cross_entropy_with_logits,
                       ops::SigmoidCrossEntropyWithLogitsXPUKernel<
                           paddle::platform::XPUDeviceContext, float>);

REGISTER_OP_XPU_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                       ops::SigmoidCrossEntropyWithLogitsGradXPUKernel<
                           paddle::platform::XPUDeviceContext, float>);

#endif
