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

#include "paddle/fluid/operators/dgc_comm_op.h"

namespace paddle {
namespace operators {

class DGCCommOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DGCCommOp");

    OP_INOUT_CHECK(ctx->HasOutput("Gather_Out"), "Output", "Gather_Out",
                   "DGCCommOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "DGCCommOp");

    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for dgc comm op must be non-negative.", ring_id));

    int nranks = ctx->Attrs().Get<int>("nranks");
    PADDLE_ENFORCE_GT(
        nranks, 1,
        platform::errors::InvalidArgument(
            "The nranks (%d) for dgc comm op must be greater than 1. ",
            ring_id));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class DGCCommOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) encoded grad with 2 * k value in dgc. ");
    AddAttr<int>("nranks",
                 "(int) the number of trainers which must be more than 1.");
    AddAttr<int>("k_var", "(int) the number of values of sparse grad. ");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);

    AddOutput("Gather_Out", "(Tensor) the gather result of dgc comm op.");
    AddOutput("Out", "(Tensor) the result of dgc comm op.");
    AddComment(R"DOC(
DGC Comm Operator
use allgather comm op to compute the grad reduce in dgc.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(dgc_comm, ops::DGCCommOp, ops::DGCCommOpMaker)

REGISTER_OP_CPU_KERNEL(
    dgc_comm,
    ops::DGCCommOpCPUKernel<paddle::platform::CPUDeviceContext, float>);
