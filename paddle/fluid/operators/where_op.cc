/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/where_op.h"

namespace paddle {
namespace operators {

class WhereOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Condition"),
                   "Input(Condition) of WhereOp should not be null.");
    PADDLE_ENFORCE(
        ctx->GetInputDim("Condition").size() >= 1,
        "Input(Condition) should have number of dimension at least 1");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(OUt) of WhereOp should not be null.");
    ctx->SetOutputDim("Out", {-1, ctx->GetInputDim("Condition").size()});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto output_type = framework::proto::VarType::INT64;
    return framework::OpKernelType(output_type, ctx.device_context());
  }
};

class WhereOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Condition", "A bool tensor whose rank is at least 1");
    AddOutput("Out", "An int64 tensor of rank 2");
    AddComment(R"DOC(
      Return a int64 tensor with rank 2, specifying the coordinate of true element in `Condition`.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(where, ops::WhereOp, ops::WhereOpMaker);
REGISTER_OP_CPU_KERNEL(where, ops::CPUWhereKernel<int64_t>);
