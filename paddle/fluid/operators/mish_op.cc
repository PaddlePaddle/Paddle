/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mish_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class MishOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mish");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "mish");

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class MishOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Mish operator");
    AddOutput("Out", "Output of Mish operator");
    AddAttr<float>(
        "threshold",
        "Constant threshold of softplus in Mish operator. Approximate value "
        "of softplus will be used if absolute value of input is greater than "
        ":attr:`threshold`")
        .SetDefault(20.f);
    AddComment(R"DOC(
Mish Activation Operator.

..  math::
    softplus = \begin{cases}
            x, \text{if } x > \text{threshold} \\
            e^{x}, \text{if } x < -\text{threshold} \\
            \ln(1 + e^{x}),  \text{otherwise}
          \end{cases}

    out = x * \tanh(softplus)

)DOC");
  }
};

// The operator to calculate gradients of a prelu operator.
class MishGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "mish");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mish");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

template <typename T>
class MishGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("mish_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(mish, ops::MishOp, ops::MishOpMaker,
                  ops::MishGradOpMaker<paddle::framework::OpDesc>,
                  ops::MishGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(mish_grad, ops::MishGradOp);
REGISTER_OP_CPU_KERNEL(
    mish, ops::MishFP32CPUKernel<paddle::platform::CPUDeviceContext>,
    ops::MishCPUKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    mish_grad, ops::MishGradFP32CPUKernel<paddle::platform::CPUDeviceContext>,
    ops::MishGradCPUKernel<paddle::platform::CPUDeviceContext, double>);
