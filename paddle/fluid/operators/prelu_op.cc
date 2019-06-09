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

#include "paddle/fluid/operators/prelu_op.h"
#include <string>

namespace paddle {
namespace operators {

class PReluOp : public framework::OperatorWithKernel {
 public:
  PReluOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    std::string mode = ctx->Attrs().Get<std::string>("mode");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of PreluOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Alpha"),
                   "Input(Alpha) of PreluOp should not be null");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PreluOp should not be null");
    if (mode == "all") {
      PADDLE_ENFORCE(product(ctx->GetInputDim("Alpha")) == 1,
                     "For mode 'all', size of weight Alpha must be one.");
    } else if (mode == "channel") {
      PADDLE_ENFORCE(product(ctx->GetInputDim("Alpha")) == x_dim[1],
                     "For channel-wise mode, size of weight Alpha must be "
                     "equal to the number of channels, should be %d",
                     x_dim[1]);
    } else if (mode == "element") {
      PADDLE_ENFORCE(product(ctx->GetInputDim("Alpha")) == product(x_dim),
                     "For element-wise mode, size of weight Alpha must be "
                     "equal to the number of input, should be %d",
                     product(x_dim));
    } else {
      PADDLE_THROW("Unkown mode %s", mode);
    }
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class PReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of prelu operator.");
    AddInput("Alpha", "The alpha weight of prelu operator.");
    AddOutput("Out", "The output tensor of prelu operator.");
    AddComment(R"DOC(
PRelu Operator.
The equation is:
$$
f(x) =
\begin{cases}
\alpha * x, \quad  \text{if} \ x < 0 \\
x,         \qquad  \text{if} \ x >= 0
\end{cases}
$$
The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input `X`.
There are modes: 
  all: all elements share same weight
  channel: elements in a channel share same weight
  element: each element has a weight 
)DOC");
    AddAttr<std::string>("mode", "The mode for inputs to share weights.")
        .SetDefault("all");
  }
};

// The operator to calculate gradients of a prelu operator.
class PReluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_grad_name = framework::GradVarName("X");
    auto alpha_grad_name = framework::GradVarName("Alpha");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(alpha_grad_name)) {
      ctx->SetOutputDim(alpha_grad_name, ctx->GetInputDim("Alpha"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   platform::CPUPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(prelu, ops::PReluOp, ops::PReluOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(prelu_grad, ops::PReluGradOp);
REGISTER_OP_CPU_KERNEL(
    prelu, ops::PReluKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    prelu_grad,
    ops::PReluGradKernel<paddle::platform::CPUDeviceContext, float>);
