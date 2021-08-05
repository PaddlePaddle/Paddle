/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ExpNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "Ensure the op execute on NPU" << std::endl;
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");

    framework::NPUAttributeMap attr_input = {};
    // set attrs if have
    if (ctx.HasAttr("base")) {
      attr_input["base"] = ctx.Attr<float>("base");
    }
    if (ctx.HasAttr("scale")) {
      attr_input["scale"] = ctx.Attr<float>("scale");
    }
    if (ctx.HasAttr("shift")) {
      attr_input["shift"] = ctx.Attr<float>("shift");
    }

    out->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("Exp", {*x}, {*out}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ExpGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    dx->mutable_data<T>(ctx.GetPlace());

    const auto& runner = NpuOpRunner("Mul", {*dout, *out}, {*dx}, {});

    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    exp, ops::ExpNPUKernel<paddle::platform::NPUDeviceContext, float>,

    ops::ExpNPUKernel<paddle::platform::NPUDeviceContext, double>);

REGISTER_OP_NPU_KERNEL(
    exp_grad, ops::ExpGradNPUKernel<paddle::platform::NPUDeviceContext, float>,

    ops::ExpGradNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::ExpGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::ExpGradNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
