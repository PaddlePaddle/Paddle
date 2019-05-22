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

#include "paddle/fluid/operators/distributed_ops/broadcast_op.h"

namespace paddle {
namespace operators {

class BroadcastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Output<framework::Tensor>("Out")->type(),
                                   ctx.GetPlace());
  }
};

class BroadcastOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("Out", "(Tensor) the result of broadcast.");
    AddAttr<bool>("sync_mode",
        "(bool) whether to synchronize data.") .SetDefault(false);
    AddAttr<int>("group", "(int) nccl communication group id.").SetDefault(0);
    AddAttr<int>("root", "(int) root id for broadcasting.").SetDefault(0);
    AddComment(R"DOC(
***Broadcast Operator***

Call ncclBcast internally.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(broadcast, ops::BroadcastOp,
                             ops::BroadcastOpMaker);

REGISTER_OP_CPU_KERNEL(
    broadcast, ops::BroadcastOpKernel<plat::CPUDeviceContext, float>,
    ops::BroadcastOpKernel<plat::CPUDeviceContext, double>,
    ops::BroadcastOpKernel<plat::CPUDeviceContext, int>,
    ops::BroadcastOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::BroadcastOpKernel<plat::CPUDeviceContext, plat::float16>);
