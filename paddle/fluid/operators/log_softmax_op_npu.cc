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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"

namespace paddle {
namespace operators {

using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
class LogSoftmaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<framework::Tensor>("X");
    auto* Out = ctx.Output<framework::Tensor>("Out");
    const int rank = X->dims().size();
    const int axis = phi::funcs::CanonicalAxis(ctx.Attr<int>("axis"), rank);
    Out->mutable_data<T>(ctx.GetPlace());

    if (X->numel() != 0) {
      auto stream = ctx.template device_context<NPUDeviceContext>().stream();
      const auto& runner = NpuOpRunner("LogSoftmaxV2", {*X}, {*Out},
                                       {{"axes", std::vector<int>{axis}}});
      runner.Run(stream);
    }
  }
};

template <typename T>
class LogSoftmaxGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* Out = ctx.Input<framework::Tensor>("Out");
    auto* dOut = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dX = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    const int rank = dOut->dims().size();
    const int axis = phi::funcs::CanonicalAxis(ctx.Attr<int>("axis"), rank);

    // allocate memory on device.
    dX->mutable_data<T>(ctx.GetPlace());

    if (dOut->numel() != 0) {
      auto stream = ctx.template device_context<NPUDeviceContext>().stream();
      const auto& runner = NpuOpRunner("LogSoftmaxGrad", {*dOut, *Out}, {*dX},
                                       {{"axis", std::vector<int>{axis}}});
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(log_softmax, ops::LogSoftmaxNPUKernel<float>,
                       ops::LogSoftmaxNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(log_softmax_grad, ops::LogSoftmaxGradNPUKernel<float>,
                       ops::LogSoftmaxGradNPUKernel<plat::float16>);
