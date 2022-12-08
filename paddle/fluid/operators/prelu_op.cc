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

#include <memory>
#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class PReluOp : public framework::OperatorWithKernel {
 public:
  PReluOp(const std::string &type,
          const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
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
    AddAttr<std::string>("data_format",
                         "Data format that specifies the layout of input")
        .SetDefault("NCHW");
  }
};

// The operator to calculate gradients of a prelu operator.
class PReluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "prelu");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "prelu");

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
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class PReluGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("prelu_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Alpha", this->Input("Alpha"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Alpha"), this->InputGrad("Alpha"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(prelu,
                            PReluInferShapeFunctor,
                            PD_INFER_META(phi::PReluInferMeta));
REGISTER_OPERATOR(prelu,
                  ops::PReluOp,
                  ops::PReluOpMaker,
                  ops::PReluGradOpMaker<paddle::framework::OpDesc>,
                  ops::PReluGradOpMaker<paddle::imperative::OpBase>,
                  PReluInferShapeFunctor);
REGISTER_OPERATOR(prelu_grad, ops::PReluGradOp);
