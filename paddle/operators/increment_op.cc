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

#include "paddle/operators/increment_op.h"

namespace paddle {
namespace operators {

class IncrementOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IncrementOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of IncrementOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class IncrementOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IncrementOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) The input tensor of increment operator");
    AddOutput("Out", "(Tensor) The output tensor of increment operator.");
    AddAttr<float>("step",
                   "(float, default 1.0) "
                   "The step size by which the "
                   "input tensor will be incremented.")
        .SetDefault(1.0);
    AddComment(R"DOC(
Increment Operator.

The equation is: 
$$Out = X + step$$

)DOC");
  }
};

class IncrementGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto *grad_op = new framework::OpDescBind();
    grad_op->SetType("scale");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttr("scale", 1.0f);
    return std::unique_ptr<framework::OpDescBind>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(increment, ops::IncrementOp, ops::IncrementOpMaker,
                  ops::IncrementGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    increment, ops::IncrementKernel<paddle::platform::CPUPlace, float>,
    ops::IncrementKernel<paddle::platform::CPUPlace, double>,
    ops::IncrementKernel<paddle::platform::CPUPlace, int>,
    ops::IncrementKernel<paddle::platform::CPUPlace, int64_t>);
