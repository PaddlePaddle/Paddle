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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class SquaredL2NormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<phi::DenseTensor>("X");
    auto *out = context.Output<phi::DenseTensor>("Out");

    auto place = context.GetPlace();
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axis;
    for (int i = 0; i < x->dims().size(); ++i) {
      axis.push_back(i);
    }
    out->mutable_data<T>(place);
    const auto &runner = NpuOpRunner(
        "SquareSumV1", {*x}, {*out}, {{"axis", axis}, {"keep_dims", false}});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SquaredL2NormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<phi::DenseTensor>("X");
    auto *x_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *out_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(
        out_grad->numel(),
        1,
        platform::errors::InvalidArgument(
            "Input(GRAD@Out) of SquaredL2NormGradOP should be a scalar."));

    auto place = context.GetPlace();
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // broadcast out_grad
    Tensor broadcasted_out_grad;
    broadcasted_out_grad.mutable_data<T>(x_grad->dims(), place);
    const auto &broadcast_runner =
        NpuOpRunner("BroadcastToD",
                    {*out_grad},
                    {broadcasted_out_grad},
                    {{"shape", phi::vectorize(x_grad->dims())}});
    broadcast_runner.Run(stream);
    // mul x
    Tensor tmp_x_grad;
    tmp_x_grad.mutable_data<T>(x_grad->dims(), place);
    const auto &mul_x_runner =
        NpuOpRunner("Mul", {broadcasted_out_grad, *x}, {tmp_x_grad}, {});
    mul_x_runner.Run(stream);
    // mul coefficient:2
    Tensor coefficient;
    coefficient.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&coefficient, static_cast<T>(2.0));
    x_grad->mutable_data<T>(place);
    const auto &mul_coefficient_runner =
        NpuOpRunner("Mul", {tmp_x_grad, coefficient}, {*x_grad}, {});
    mul_coefficient_runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    squared_l2_norm,
    ops::SquaredL2NormNPUKernel<plat::NPUDeviceContext, float>);
REGISTER_OP_NPU_KERNEL(
    squared_l2_norm_grad,
    ops::SquaredL2NormGradNPUKernel<plat::NPUDeviceContext, float>);
