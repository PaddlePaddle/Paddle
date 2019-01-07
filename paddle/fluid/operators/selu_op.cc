/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/selu_op.h"
#include <string>

namespace paddle {
namespace operators {

class SeluOp : public framework::OperatorWithKernel {
 public:
  SeluOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SeluOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SeluOp should not be null.");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::GetDataTypeOfVar(ctx.InputVar("X")), ctx.GetPlace());
  }
};

class SeluOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

class SeluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of selu operator.");
    AddOutput("Out", "The output tensor of selu operator.");
    AddAttr<float>("scale",
                   "(float) the default value is 1.0507~. For more "
                   "information about this value, please refer to:"
                   "https://arxiv.org/abs/1706.02515.")
        .SetDefault(1.0507009873554804934193349852946);
    AddAttr<float>("alpha",
                   "(float) the default value is 1.6732~. For more "
                   "information about this value, please refer to:"
                   "https://arxiv.org/abs/1706.02515.")
        .SetDefault(1.6732632423543772848170429916717);
    AddComment(R"DOC(
Selu Operator.

The equation is:
$$
f(x) =\lambda*
\begin{cases}
 \quad \quad   x,  \quad \quad \quad \text{if} \ x > 0 \\
 \alpha * e^x - \alpha,  \qquad  \text{if} \ x <= 0
\end{cases}
$$

The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input `X`.
)DOC");
  }
};

class SeluGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("selu_grad");
    grad_op->SetInput("Out", Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class SeluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Out"), "Input(Out) should not be null");
    auto x_grad_name = framework::GradVarName("X");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("Out"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::GetDataTypeOfVar(ctx.InputVar("Out")), ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(selu, ops::SeluOp, ops::SeluOpMaker, ops::SeluOpInferVarType,
                  ops::SeluGradMaker);
REGISTER_OPERATOR(selu_grad, ops::SeluGradOp);
REGISTER_OP_CPU_KERNEL(
    selu, ops::SeluKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SeluKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    selu_grad, ops::SeluGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SeluGradKernel<paddle::platform::CPUDeviceContext, double>);
