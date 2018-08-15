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

#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class GatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of GatherOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Index"),
                   "Input(Index) of GatherOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GatherOp should not be null.");

    auto index_dims = ctx->GetInputDim("Index");
    PADDLE_ENFORCE(index_dims.size() == 1);
    int batch_size = ctx->GetInputDim("Index")[0];
    framework::DDim output_dims(ctx->GetInputDim("X"));
    output_dims[0] = batch_size;
    ctx->SetOutputDim("Out", output_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class GatherGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class GatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of gather op");
    AddInput("Index", "The index input of gather op");
    AddOutput("Out", "The output of gather op");
    AddComment(R"DOC(
Gather Operator.

$Out = X[Index]$

Out is obtained by gathering entries of the outer-most dimension 
of X indexed by Index and concatenate them together.

Example:

X = [[1, 2],
     [3, 4],
     [5, 6]]

Index = [[1, 2]]

Then:

Out = [[3, 4],
       [5, 6]]

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gather, ops::GatherOp, ops::GatherOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(gather_grad, ops::GatherGradOp);
REGISTER_OP_CPU_KERNEL(gather, ops::GatherOpKernel<float>);
REGISTER_OP_CPU_KERNEL(gather_grad, ops::GatherGradientOpKernel<float>);
