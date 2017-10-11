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

#include "paddle/operators/prelu_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class PReluOp : public framework::OperatorWithKernel {
 public:
  PReluOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Alpha"), "Input(Alpha) should not be null");
    PADDLE_ENFORCE(product(ctx->GetInputDim("Alpha")) == 1,
                   "Size of weight Alpha must be one.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class PReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PReluOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of prelu operator.");
    AddInput("Alpha", "The alpha weight of PRelu operator.");
    AddOutput("Out", "The output tensor of PRelu operator.");
    AddComment(R"DOC(PRelu operator

The equation is:

  f(x) = alpha * x , for x < 0
  f(x) = x         , for x >= 0

The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD with input `X`.
)DOC");
  }
};

// The operator to calculate gradients of a prelu operator.
class PReluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("Alpha"),
                      ctx->GetInputDim("Alpha"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(prelu, ops::PReluOp, ops::PReluOpMaker, prelu_grad,
            ops::PReluGradOp);
REGISTER_OP_CPU_KERNEL(prelu,
                       ops::PReluKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(prelu_grad,
                       ops::PReluGradKernel<paddle::platform::CPUPlace, float>);
