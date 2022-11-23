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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class GeluNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");

    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Gelu", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class GeluGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // NOTE(pangyoki): In the original implementation of GeluGrad op, the input
    // is {*dout, *x, out}, where out = Gelu(x). However, we find that variable
    // `out` was not actually used. In order to improve performance, the
    // useless GELU operation was deleted.
    // We directly use `*dout` as a placeholder to replace `out`, it will not
    // be used in calculations.
    const auto& runner_dx =
        NpuOpRunner("GeluGrad", {*dout, *x, *dout}, {*dx}, {});
    runner_dx.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    gelu,
    ops::GeluNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GeluNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    gelu_grad,
    ops::GeluGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::GeluGradNPUKernel<paddle::platform::NPUDeviceContext,
                           paddle::platform::float16>);
