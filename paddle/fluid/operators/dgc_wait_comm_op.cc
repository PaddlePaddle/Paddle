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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dgc_wait_comm_op.h"

namespace paddle {
namespace operators {

class DGCWaitCommOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for dgc comm op must be non-negative.", ring_id));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class DGCWaitCommOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddOutput("Out", "(Tensor) Dependency of the variable need to sync")
        .AsDuplicable();
    AddAttr<int>("ring_id", "(int default 0) ring id.").SetDefault(0);
    AddComment(R"DOC(
CSyncCommStream Operator
Call communication stream synchronization.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_WITHOUT_GRADIENT(dgc_wait_comm, ops::DGCWaitCommOp,
                             ops::DGCWaitCommOpMaker)

REGISTER_OP_CPU_KERNEL(
    dgc_wait_comm,
    ops::DGCWaitCommOpCPUKernel<phi::CPUContext, float>);
