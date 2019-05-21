/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/operators/distributed_ops/nccl_broadcast_op.h"

namespace paddle {
namespace operators {

class NCCLBroadcastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class NCCLBroadcastOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be broadcast.");
    AddOutput("Out", "(Tensor) the result of broadcast.");
    AddAttr<bool>(
        "sync_mode",
        "(bool) whether to synchronize the CUDA stream after nccl call.")
        .SetDefault(false);
    AddComment(R"DOC(
***NCCLBroadcast Operator***

Call NCCLBroadcast internally.
If input and output are the same variable, in-place broadcast will be used.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(nccl_broadcast, ops::NCCLBroadcastOp,
                             ops::NCCLBroadcastOpMaker);

REGISTER_OP_CPU_KERNEL(
    nccl_broadcast, ops::NCCLBroadcastOpKernel<plat::CPUDeviceContext, float>,
    ops::NCCLBroadcastOpKernel<plat::CPUDeviceContext, double>,
    ops::NCCLBroadcastOpKernel<plat::CPUDeviceContext, int>,
    ops::NCCLBroadcastOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::NCCLBroadcastOpKernel<plat::CPUDeviceContext, plat::float16>);
