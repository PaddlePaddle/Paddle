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

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/operators/erf_op.h"

#define _USE_MATH_DEFINES

namespace paddle {
namespace operators {

class ErfOp : public framework::OperatorWithKernel {
 public:
  ErfOp(const std::string &type, const framework::VariableNameMap &inputs,
        const framework::VariableNameMap &outputs,
        const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of ErfOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ErfOp should not be null.");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class ErfOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

class ErfGradOp : public framework::OperatorWithKernel {
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
        OperatorWithKernel::IndicateVarDataType(ctx, "Out"), ctx.GetPlace());
  }
};

class ErfOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of erf operator.");
    AddOutput("Out", "The output tensor of erf operator.");
    AddComment(R"DOC(
Erf Operator.

The equation is:
$$
f(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta
$$

The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input `X`.
)DOC");
  }
};

template <typename T>
class ErfGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("erf_grad");
    grad_op->SetInput("X", this->Output("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(erf, ops::ErfOp, ops::ErfOpMaker, ops::ErfOpInferVarType,
                  ops::ErfGradOpMaker<paddle::framework::OpDesc>,
                  ops::ErfGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(ref_grad, ops::ErfGradOp);
REGISTER_OP_CPU_KERNEL(
    erf, ops::ErfKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ErfKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    erf_grad, ops::ErfGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ErfGradKernel<paddle::platform::CPUDeviceContext, double>);
