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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

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
    int ignore_index = context.Attr<int>("ignore_index");
    bool normalize = context.Attr<bool>("normalize");

    // allocate temp memory
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int* hit = RAII_GUARD.alloc_l3_or_gm<int>(input->numel());
    PADDLE_ENFORCE_NOT_NULL(
        hit, platform::errors::External("XPU alloc_l3_or_gm returns nullptr"));

    int r = xpu::sigmoid_cross_entropy_with_logits(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(input->data<T>()),
        reinterpret_cast<const XPUType*>(label->data<T>()),
        reinterpret_cast<XPUType*>(output->data<T>()), 1, input->numel(), hit,
        ignore_index);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sigmoid_cross_entropy_with_logits");
    if (normalize) {
      int* non_zero = RAII_GUARD.alloc_l3_or_gm<int>(1);
      PADDLE_ENFORCE_NOT_NULL(
          non_zero,
          platform::errors::External("XPU alloc_l3_or_gm returns nullptr"));
      int r = xpu::nonzero_count(dev_ctx.x_context(),
                                 reinterpret_cast<const XPUType*>(hit),
                                 non_zero, input->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "nonzero_count");
      int non_zero_cpu = 0;
      memory::Copy(platform::CPUPlace(), static_cast<void*>(&non_zero_cpu),
                   context.GetPlace(), static_cast<void*>(non_zero),
                   sizeof(int));
      r = xpu::scale(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(output->data<T>()),
                     reinterpret_cast<XPUType*>(output->data<T>()),
                     input->numel(), false,
                     1.0f / static_cast<float>(non_zero_cpu), 0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
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

    // attrs
    int ignore_index = context.Attr<int>("ignore_index");
    bool normalize = context.Attr<bool>("normalize");

    // allocate temp memory
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int* hit = RAII_GUARD.alloc_l3_or_gm<int>(input->numel());
    PADDLE_ENFORCE_NOT_NULL(
        hit, platform::errors::External("XPU alloc_l3_or_gm returns nullptr"));

    int r = xpu::sigmoid_cross_entropy_with_logits_grad(
        dev_ctx.x_context(), reinterpret_cast<const XPUType*>(input->data<T>()),
        reinterpret_cast<const XPUType*>(label->data<T>()),
        reinterpret_cast<const XPUType*>(dy->data<T>()),
        reinterpret_cast<XPUType*>(dx->data<T>()), 1, input->numel(), hit,
        ignore_index);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sigmoid_cross_entropy_with_logits");
    if (normalize) {
      int* non_zero = RAII_GUARD.alloc_l3_or_gm<int>(1);
      PADDLE_ENFORCE_NOT_NULL(
          non_zero,
          platform::errors::External("XPU alloc_l3_or_gm returns nullptr"));
      int r = xpu::nonzero_count(dev_ctx.x_context(),
                                 reinterpret_cast<const XPUType*>(hit),
                                 non_zero, input->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "nonzero_count");
      int non_zero_cpu = 0;
      memory::Copy(platform::CPUPlace(), static_cast<void*>(&non_zero_cpu),
                   context.GetPlace(), static_cast<void*>(non_zero),
                   sizeof(int));
      r = xpu::scale(dev_ctx.x_context(),
                     reinterpret_cast<const XPUType*>(dx->data<T>()),
                     reinterpret_cast<XPUType*>(dx->data<T>()), input->numel(),
                     false, 1.0f / static_cast<float>(non_zero_cpu), 0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
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
