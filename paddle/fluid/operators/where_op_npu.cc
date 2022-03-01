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

#include "paddle/fluid/operators/where_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class WhereNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* condition = ctx.Input<framework::Tensor>("Condition");
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    const auto& runner =
        NpuOpRunner("Select", {*condition, *X, *Y}, {*out}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class WhereGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* condition = ctx.Input<framework::Tensor>("Condition");
    auto* dout_t = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx_t = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy_t = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

    if (dx_t != nullptr) {
      dx_t->mutable_data<T>(ctx.GetPlace());
    }
    if (dy_t != nullptr) {
      dy_t->mutable_data<T>(ctx.GetPlace());
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    framework::Tensor tensor_zeros(dout_t->dtype());
    tensor_zeros.mutable_data<T>(dout_t->dims(), ctx.GetPlace());
    const auto& runner =
        NpuOpRunner("ZerosLike", {*dout_t}, {tensor_zeros}, {});
    runner.Run(stream);

    if (dx_t != nullptr) {
      const auto& runner = NpuOpRunner(
          "Select", {*condition, *dout_t, tensor_zeros}, {*dx_t}, {});
      runner.Run(stream);
    }
    if (dy_t != nullptr) {
      const auto& runner = NpuOpRunner(
          "Select", {*condition, tensor_zeros, *dout_t}, {*dy_t}, {});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    where, ops::WhereNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::WhereNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::WhereNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::WhereNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(
    where_grad,
    ops::WhereGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::WhereGradNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::WhereGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::WhereGradNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
