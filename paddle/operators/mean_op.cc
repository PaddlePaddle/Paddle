/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/mean_op.h"

namespace paddle {
namespace operators {

class MeanOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MeanOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MeanOp should not be null.");
    ctx->SetOutputDim("Out", {1});
  }
};

class MeanOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MeanOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of mean op");
    AddOutput("Out", "The output of mean op");
    AddComment(R"DOC(
Mean Operator.

Out is a scalar which is the mean of all elements in X. 

)DOC");
  }
};

class MeanGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }
};

class MeanGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto* grad_op = new framework::OpDescBind();
    grad_op->SetType("mean_grad");
    grad_op->SetInput("X", Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    return std::unique_ptr<framework::OpDescBind>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mean, ops::MeanOp, ops::MeanOpMaker, ops::MeanGradMaker);
REGISTER_OPERATOR(mean_grad, ops::MeanGradOp);
REGISTER_OP_CPU_KERNEL(mean, ops::MeanKernel<paddle::platform::CPUPlace, float>,
                       ops::MeanKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(mean_grad,
                       ops::MeanGradKernel<paddle::platform::CPUPlace, float>,
                       ops::MeanGradKernel<paddle::platform::CPUPlace, double>);
