// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"
namespace paddle {
namespace operators {

class WhereOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class WhereGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Condition"), "Input", "Condition", "Where");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Where");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "Where");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "Where");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

class WhereOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Condition",
             "(Tensor) A bool tensor whose rank is at least 1. When Condition "
             "is True, yield x, otherwise yield y");
    AddInput("X",
             "(Tensor), The first input tensor of where op. When the "
             "corresponding position of the condition is true, the output "
             "takes the element of X.");
    AddInput("Y",
             "(Tensor), The second input tensor of where op. When the "
             "corresponding position of condition is false, the output takes "
             "the element of Y.");
    AddOutput("Out", "(Tensor), The output tensor of where op.");
    AddComment(R"DOC(
      Where Operator.
      Return a tensor of elements selected from either $X$ or $Y$, depending on condition.
      The equation is:
      $$
      Out_i =
      \begin{cases}
      \X_i, \quad  \text{if} \ cond_i is True \\
      \Y_i, \quad  \text{if} \ cond_i is False \\
      \end{cases}
      $$
)DOC");
  }
};

template <typename T>
class WhereOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("where_grad");
    grad->SetInput("Condition", this->Input("Condition"));
    grad->SetInput("X", this->Input("X"));
    grad->SetInput("Y", this->Input("Y"));
    grad->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(WhereGradNoNeedBufferVarsInferer, "X", "Y");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(where, WhereInferShapeFunctor,
                            PD_INFER_META(phi::WhereInferMeta));
REGISTER_OPERATOR(where, ops::WhereOp, ops::WhereOpMaker,
                  ops::WhereOpGradMaker<paddle::framework::OpDesc>,
                  ops::WhereOpGradMaker<paddle::imperative::OpBase>,
                  WhereInferShapeFunctor);

REGISTER_OPERATOR(where_grad, ops::WhereGradOp,
                  ops::WhereGradNoNeedBufferVarsInferer);
