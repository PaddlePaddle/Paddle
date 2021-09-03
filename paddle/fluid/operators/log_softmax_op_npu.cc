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

#include "paddle/fluid/operators/log_softmax_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class LogSoftmaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Out = ctx.Output<framework::Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    std::vector<int> axes;
    axes.push_back(axis);
    framework::NPUAttributeMap attr_input = {{"axes", axes}};
    Out->mutable_data<T>(ctx.GetPlace());
    const auto& runner = NpuOpRunner("LogSoftmaxV2", {*X}, {*Out}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    log_softmax,
    ops::LogSoftmaxNPUKernel<paddle::platform::NPUDeviceContext, float>);
