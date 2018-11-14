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

#include "paddle/fluid/operators/minus_op.h"

#include <string>
#include <vector>

namespace paddle {
namespace operators {

class MinusOp : public framework::OperatorWithKernel {
 public:
  MinusOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MinusOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of MinusOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MinusOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(
        x_dims, y_dims,
        "Minus operator must take two tensor with same num of elements");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MinusOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The left tensor of minus operator.");
    AddInput("Y", "The right tensor of minus operator.");
    AddOutput("Out", "The output tensor of minus operator.");

    AddComment(R"DOC(
Minus Operator.

Equation:

    $Out = X - Y$

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `X`.

)DOC");
  }
};

class MinusGradMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    auto x_g = InputGrad("X");
    if (!x_g.empty()) {
      auto *x_g_op = new framework::OpDesc();
      x_g_op->SetType("scale");
      x_g_op->SetInput("X", OutputGrad("Out"));
      x_g_op->SetOutput("Out", x_g);
      x_g_op->SetAttr("scale", 1.0f);
      ops.emplace_back(x_g_op);
    }

    auto y_g = InputGrad("Y");
    if (!y_g.empty()) {
      auto *y_g_op = new framework::OpDesc();
      y_g_op->SetType("scale");
      y_g_op->SetInput("X", OutputGrad("Out"));
      y_g_op->SetOutput("Out", y_g);
      y_g_op->SetAttr("scale", -1.0f);
      ops.emplace_back(y_g_op);
    }

    return ops;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(minus, ops::MinusOp, ops::MinusOpMaker, ops::MinusGradMaker);
REGISTER_OP_CPU_KERNEL(
    minus, ops::MinusKernel<paddle::platform::CPUDeviceContext, float>);
