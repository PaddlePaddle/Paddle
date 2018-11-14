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
    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of SeluOp should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SeluOp should not be null");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::GetDataTypeOfVar(ctx.InputVar("X")), platform::CPUPlace());
  }
};

class SeluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of selu operator.");
    AddOutput("Out", "The output tensor of selu operator.");
    AddAttr<float>(
        "alpha",
        "(float) default to 1.6732~; affects the activation function itself. "
        "This should go with the weight initialization in the paper. "
        " See https://arxiv.org/abs/1706.02515 ")
        .SetDefault(1.6732632423543772848170429916717);
    AddAttr<float>(
        "scale",
        "(float) default to 1.0507~; affects the activation function itself.")
        .SetDefault(1.0507009873554804934193349852946);
    AddComment(R"DOC(
Selu Operator.

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function, y = scale*(alpha_*e^x-alpha_ if x < 0 else x),
is applied to the tensor elementwise.

)DOC");
    AddAttr<std::string>("mode", "The mode for inputs to share weights.")
        .SetDefault("all");
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
// The operator to calculate gradients of a selu operator.
class SeluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_grad_name = framework::GradVarName("X");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::GetDataTypeOfVar(ctx.InputVar("Out")), platform::CPUPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(selu, ops::SeluOp, ops::SeluOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(selu_grad, ops::SeluGradOp);
REGISTER_OP_CPU_KERNEL(
    selu, ops::SeluKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    selu_grad, ops::SeluGradKernel<paddle::platform::CPUDeviceContext, float>);
