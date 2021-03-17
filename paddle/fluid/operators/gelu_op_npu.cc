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

#include "paddle/fluid/operators/gelu_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class GeluNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    auto runner = NpuOpRunner("Gelu", {*x}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class GeluGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor out(x->type());
    out.mutable_data<T>(x->dims(), place);
    auto out_runner = NpuOpRunner("Gelu", {*x}, {out}, {});
    out_runner.Run(stream);

    auto dx_runner = NpuOpRunner("GeluGrad", {*dout, *x, out}, {*dx}, {});
    dx_runner.Run(stream);
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
