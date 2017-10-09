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

#pragma once
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {

class ElementwiseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  using Tensor = framework::Tensor;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of elementwise op should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of elementwise op should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of elementwise op should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                      "Rank of first input must >= rank of second input.")
    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ElementwiseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ElementwiseOpMaker(framework::OpProto* proto,
                     framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", R"DOC(
The first input of elementwise op, it's a tensor of any dimensions.
)DOC");
    AddInput("Y", R"DOC(
The sencond input of elementwise op, it's a tensor and it's dimensions
must be small or equal to X's dimensions.
)DOC");
    AddAttr<int>("axis",
                 R"DOC(
When the shape(Y) does not equal the shape(X),Y will be broadcasted
to match the shape of X and axis should be dimension index Y in X
        )DOC")
        .SetDefault(-1)
        .EqualGreaterThan(-1);

    AddOutput("Out", "The output of elementwise op");
    comment_ = R"DOC(
Limited elementwise {name} operator.The equation is: Out = {equation}.
1. The shape of Y should be same with X or
2. Y's shape is a subset of X.
   Y will be broadcasted to match the shape of X and axis should be dimension index Y in X.

   example:
      shape(X) = (2, 3, 4, 5), shape(Y) = (,)
      shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
      shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5)
      shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
      shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0

Both the input X and Y can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with input X.
)DOC";
    AddComment(comment_);
  }

 protected:
  std::string comment_;

  void Replace(std::string& src, std::string from, std::string to) {
    std::size_t len_from = std::strlen(from.c_str());
    std::size_t len_to = std::strlen(to.c_str());
    for (std::size_t pos = src.find(from); pos != std::string::npos;
         pos = src.find(from, pos + len_to)) {
      src.replace(pos, len_from, to);
    }
  }

  void SetComment(std::string name, std::string equation) {
    Replace(comment_, "{name}", name);
    Replace(comment_, "{equation}", equation);
  }
};

class ElementwiseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Rank of first input must >= rank of second input.")

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
