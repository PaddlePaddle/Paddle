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

// TODO(Aganlengzi): delete this macro control and remove REMOVE_ITEM in
// cmake/operators.cmake when Paddle supports
#if (CANN_VERSION_CODE >= 504000)

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class NPUTakeAlongAxisKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("Input");
    auto axis = ctx.Attr<int>("Axis");
    auto index = ctx.Input<phi::DenseTensor>("Index");
    auto result = ctx.Output<phi::DenseTensor>("Result");
    result->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner(
        "GatherElements", {*input, *index}, {*result}, {{"dim", axis}});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class NPUTakeAlongAxisGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto axis = ctx.Attr<int>("Axis");
    auto index = ctx.Input<phi::DenseTensor>("Index");
    auto result_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Result"));

    auto input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    input_grad->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("ScatterAddWithAxis",
                                     {*input_grad, *index, *result_grad},
                                     {*input_grad},
                                     {{"axis", axis}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    take_along_axis,
    ops::NPUTakeAlongAxisKernel<paddle::platform::NPUDeviceContext, int>,
    ops::NPUTakeAlongAxisKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::NPUTakeAlongAxisKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NPUTakeAlongAxisKernel<paddle::platform::NPUDeviceContext, double>)
REGISTER_OP_NPU_KERNEL(
    take_along_axis_grad,
    ops::NPUTakeAlongAxisGradKernel<paddle::platform::NPUDeviceContext, int>,
    ops::NPUTakeAlongAxisGradKernel<paddle::platform::NPUDeviceContext,
                                    int64_t>,
    ops::NPUTakeAlongAxisGradKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NPUTakeAlongAxisGradKernel<paddle::platform::NPUDeviceContext, double>)

#endif
