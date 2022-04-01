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

#include <memory>
#include <string>
#include <utility>
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
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of MinusOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound("Input(Y) of MinusOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output(Out) of MinusOp is not found."));

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    if (ctx->IsRuntime() ||
        (phi::product(x_dims) > 0 && phi::product(y_dims) > 0)) {
      PADDLE_ENFORCE_EQ(
          x_dims, y_dims,
          platform::errors::InvalidArgument(
              "Minus operator must take two tensor with same dim, but received "
              "input X dim is:[%s], Y dim is:[%s]",
              x_dims, y_dims));
    }
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

class MinusGradDescMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    auto x_g = this->InputGrad("X");
    if (!x_g.empty()) {
      auto *x_g_op = new framework::OpDesc();
      x_g_op->SetType("scale");
      x_g_op->SetInput("X", this->OutputGrad("Out"));
      x_g_op->SetOutput("Out", x_g);
      x_g_op->SetAttr("scale", 1.0f);
      ops.emplace_back(x_g_op);
    }

    auto y_g = this->InputGrad("Y");
    if (!y_g.empty()) {
      auto *y_g_op = new framework::OpDesc();
      y_g_op->SetType("scale");
      y_g_op->SetInput("X", this->OutputGrad("Out"));
      y_g_op->SetOutput("Out", y_g);
      y_g_op->SetAttr("scale", -1.0f);
      ops.emplace_back(y_g_op);
    }

    return ops;
  }
};

class MinusGradMaker : public imperative::GradOpBaseMakerBase {
 public:
  using imperative::GradOpBaseMakerBase::GradOpBaseMakerBase;

  std::shared_ptr<imperative::GradOpNode> operator()() const override {
    auto x_g = this->InputGrad("X");
    auto y_g = this->InputGrad("Y");

    auto node = this->NewGradNode();

    if (!x_g.empty()) {
      imperative::TracedGradOp op(node);
      op.SetType("scale");
      op.SetInput("X", this->OutputGrad("Out"));
      op.SetOutput("Out", x_g);
      op.SetAttr("scale", 1.0f);
    }

    if (!y_g.empty()) {
      imperative::TracedGradOp op(node);
      op.SetType("scale");
      op.SetInput("X", this->OutputGrad("Out"));
      op.SetOutput("Out", y_g);
      op.SetAttr("scale", -1.0f);
    }

    return node;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(minus, ops::MinusOp, ops::MinusOpMaker,
                  ops::MinusGradDescMaker, ops::MinusGradMaker);
REGISTER_OP_CPU_KERNEL(
    minus, ops::MinusKernel<paddle::platform::CPUDeviceContext, float>);

REGISTER_OP_CUDA_KERNEL(
    minus, ops::MinusKernel<paddle::platform::CUDADeviceContext, float>);
