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

#include "paddle/fluid/operators/index_select_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class IndexSelectNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto dim = ctx.Attr<int>("dim");

    auto *out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner runner;
    runner.SetType("GatherV2")
        .AddInput(*x)
        .AddInput(*index)
        .AddInput(std::vector<int32_t>{dim})
        .AddOutput(*out);
    runner.Run(stream);
  }
};

// todo: add class 'IndexSelectGradNPUKernel' here.

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    index_select,
    ops::IndexSelectNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::IndexSelectNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::IndexSelectNPUKernel<paddle::platform::NPUDeviceContext, int64_t>);
// todo: register npu index_select_grad kernel here.
