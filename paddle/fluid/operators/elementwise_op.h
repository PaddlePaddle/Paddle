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

#pragma once
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class ElementwiseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  using Tensor = framework::Tensor;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of elementwise op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of elementwise op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of elementwise op should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                      "Rank of first input must >= rank of second input.");
    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ElementwiseOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto x_name = op_desc.Input("X")[0];
    auto out_name = op_desc.Output("Out")[0];
    auto& x = block->FindRecursiveOrCreateVar(x_name);
    auto& out = block->FindRecursiveOrCreateVar(out_name);
    out.SetType(x.GetType());
  }
};

class ElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X", "(Tensor), The first input tensor of elementwise op.");
    AddInput("Y", "(Tensor), The second input tensor of elementwise op.");
    AddOutput("Out", "The output of elementwise op.").Reuse("X");
    AddAttr<int>("axis",
                 "(int, default -1). The start dimension index "
                 "for broadcasting Y onto X.")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddComment(string::Sprintf(R"DOC(
Limited Elementwise %s Operator

The equation is:

$$%s$$

- $X$: a tensor of any dimension. 
- $Y$: a tensor whose dimensions must be less than or equal to the dimensions of $X$.

There are two cases for this operator:

1. The shape of $Y$ is the same with $X$.
2. The shape of $Y$ is a continuous subsequence of $X$.

For case 2:

1. Broadcast $Y$ to match the shape of $X$, where $axis$ is the start dimension index 
   for broadcasting $Y$ onto $X$. 
2. If $axis$ is -1 (default), $axis = rank(X) - rank(Y)$.
3. The trailing dimensions of size 1 for $Y$ will be ignored for the consideration of 
   subsequence, such as shape(Y) = (2, 1) => (2).

For example:

  .. code-block:: python

    shape(X) = (2, 3, 4, 5), shape(Y) = (,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
    shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
    shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
    shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
    shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

The inputs $X$ and $Y$ can carry the different LoD information. 
But the output only shares the LoD information with the input $X$.

)DOC",
                               GetName(), GetEquation()));
  }

 protected:
  virtual std::string GetName() const = 0;
  virtual std::string GetEquation() const = 0;
};

class ElementwiseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};
}  // namespace operators
}  // namespace paddle

#define REGISTER_ELEMWISE_OP(op_type, op_name, equation)                \
  class __ElemwiseOp##op_type##Maker__                                  \
      : public ::paddle::operators::ElementwiseOpMaker {                \
   protected:                                                           \
    virtual std::string GetName() const { return op_name; }             \
    virtual std::string GetEquation() const { return equation; }        \
  };                                                                    \
  REGISTER_OPERATOR(op_type, ::paddle::operators::ElementwiseOp,        \
                    __ElemwiseOp##op_type##Maker__,                     \
                    ::paddle::operators::ElementwiseOpInferVarType,     \
                    ::paddle::framework::DefaultGradOpDescMaker<true>); \
  REGISTER_OPERATOR(op_type##_grad, ::paddle::operators::ElementwiseOpGrad)
