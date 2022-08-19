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
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/kernels/instance_norm_grad_kernel.h"
#include "paddle/phi/kernels/instance_norm_kernel.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class InstanceNormXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("SavedMean");
    auto* variance = ctx.Output<Tensor>("SavedVariance");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // call phi kernel
    phi::InstanceNormKernel<T>(
        static_cast<const typename paddle::framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *x,
        *scale,
        *bias,
        epsilon,
        y,
        mean,
        variance);
  }
};
template <typename DeviceContext, typename T>
class InstanceNormGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto* mean = ctx.Input<Tensor>("SavedMean");
    const auto* variance = ctx.Input<Tensor>("SavedVariance");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // call phi kernel
    phi::InstanceNormGradKernel<T>(
        static_cast<const typename paddle::framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *x,
        *dy,
        *scale,
        *mean,
        *variance,
        epsilon,
        dx,
        dbias,
        dscale);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    instance_norm,
    ops::InstanceNormXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU}
