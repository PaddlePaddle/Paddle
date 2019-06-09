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

#include "paddle/fluid/operators/is_empty_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class IsEmptyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IsEmptyOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of IsEmptyOp should not be null.");
    ctx->SetOutputDim("Out", {1});
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    return framework::OpKernelType(x->type(), x->place());
  }
};

class IsEmptyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) Tensor which is to be checked.");
    AddOutput("Out",
              "(LoDTensor) a boolean Tensor that indicate empty or not.");
    AddComment(R"DOC(
IsEmpty Operator which checks whether a tensor is empty.

It will just return product(tensor.ddims()) > 0;
              )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(is_empty, ops::IsEmptyOp, ops::IsEmptyOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    is_empty, ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
