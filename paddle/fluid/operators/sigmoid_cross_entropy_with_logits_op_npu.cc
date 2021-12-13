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

#include "paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

void CheckAttrs(const framework::ExecutionContext& ctx) {
  // Add this check is is due to Ascend SigmoidCrossEntropyWithLogits
  // and SigmoidCrossEntropyWithLogitsGrad does't supoort
  // attr normalize and ignore_index
  bool normalize = ctx.Attr<bool>("normalize");
  int ignore_index = ctx.Attr<int>("ignore_index");
  PADDLE_ENFORCE_EQ(normalize, false,
                    platform::errors::InvalidArgument(
                        "attr normalize must be false, but got true"));
  PADDLE_ENFORCE_EQ(ignore_index, kIgnoreIndex,
                    platform::errors::InvalidArgument(
                        "attr ignore_index must be default %d, but got %d",
                        kIgnoreIndex, ignore_index));
}

template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    CheckAttrs(ctx);

    auto* x = ctx.Input<Tensor>("X");
    auto* label = ctx.Input<Tensor>("Label");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner =
        NpuOpRunner("SigmoidCrossEntropyWithLogits", {*x, *label}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SigmoidCrossEntropyWithLogitsNPUGradKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    CheckAttrs(ctx);

    auto* x = ctx.Input<Tensor>("X");
    auto* label = ctx.Input<Tensor>("Label");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner_dx = NpuOpRunner("SigmoidCrossEntropyWithLogitsGrad",
                                        {*x, *label, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    sigmoid_cross_entropy_with_logits,
    ops::SigmoidCrossEntropyWithLogitsNPUKernel<plat::NPUDeviceContext, float>,
    ops::SigmoidCrossEntropyWithLogitsNPUKernel<plat::NPUDeviceContext,
                                                plat::float16>);
REGISTER_OP_NPU_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    ops::SigmoidCrossEntropyWithLogitsNPUGradKernel<plat::NPUDeviceContext,
                                                    float>,
    ops::SigmoidCrossEntropyWithLogitsNPUGradKernel<plat::NPUDeviceContext,
                                                    plat::float16>);
