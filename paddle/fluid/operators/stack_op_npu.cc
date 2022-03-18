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

using Tensor = framework::Tensor;

template <typename T>
class StackNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += (x[0]->dims().size() + 1);
    int num = static_cast<int>(x.size());

    PADDLE_ENFORCE_GT(num, 0, platform::errors::InvalidArgument(
                                  "number of input Tensor <= 0"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<paddle::framework::Tensor> x_list;
    for (int i = 0; i < num; i++) {
      x_list.push_back(*x[i]);
    }
    y->mutable_data<T>(ctx.GetPlace());

    const auto& runner =
        NpuOpRunner("Pack", {x_list}, {*y}, {{"axis", axis}, {"N", num}});
    runner.Run(stream);
  }
};

template <typename T>
class StackGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();
    int num = dy->dims()[axis];

    PADDLE_ENFORCE_GT(num, 0, platform::errors::InvalidArgument(
                                  "number of input Tensor <= 0"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<paddle::framework::Tensor> dx_list;
    for (int i = 0; i < num; i++) {
      dx[i]->mutable_data<T>(ctx.GetPlace());
      dx_list.push_back(*dx[i]);
    }

    const auto& runner =
        NpuOpRunner("Unpack", {*dy}, {dx_list}, {{"axis", axis}, {"num", num}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_NPU_KERNEL(
    stack, paddle::operators::StackNPUKernel<int>,
#ifdef PADDLE_WITH_ASCEND_INT64
    paddle::operators::StackNPUKernel<int64_t>,
#endif
    paddle::operators::StackNPUKernel<float>,
    paddle::operators::StackNPUKernel<paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    stack_grad, paddle::operators::StackGradNPUKernel<int>,
#ifdef PADDLE_WITH_ASCEND_INT64
    paddle::operators::StackGradNPUKernel<int64_t>,
#endif
    paddle::operators::StackGradNPUKernel<float>,
    paddle::operators::StackGradNPUKernel<paddle::platform::float16>);
