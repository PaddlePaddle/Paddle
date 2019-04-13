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

#include "paddle/fluid/operators/modified_huber_loss_op.h"

namespace paddle {
namespace operators {

class ModifiedHuberLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X must be initialized.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Y must be initialized.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The tensor rank of X must be 2.");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims, y_dims,
                        "The shape of X and Y must be the same.");
    } else {
      if (x_dims[0] != -1 && y_dims[0] != -1) {
        PADDLE_ENFORCE_EQ(x_dims[0], y_dims[0],
                          "The dim 0 of X and Y must be the same.");
      }

      if (x_dims[1] != -1 && y_dims[1] != -1) {
        PADDLE_ENFORCE_EQ(x_dims[1], y_dims[1],
                          "The dim 1 of X and Y must be the same.");
      }
    }

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims[1], 1, "The 2nd dimension of X must be 1.");
    }

    ctx->SetOutputDim("IntermediateVal", x_dims);
    ctx->SetOutputDim("Out", {x_dims[0], 1});
  }
};

class ModifiedHuberLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of modified huber loss op. "
             "X is 2-D tensor with shape [batch_size, 1].");
    AddInput("Y",
             "The target labels of modified huber loss op. "
             "The shape of Y is the same as X. Values of Y must be 0 or 1.");
    AddOutput("IntermediateVal",
              "Variable to save intermediate result which will be reused in "
              "backward processing.")
        .AsIntermediate();
    AddOutput("Out", "Classification loss for X.");
    AddComment(R"DOC(
Modified Huber Loss Operator.

This operator is used in binary classification problem. The shape of
input X and target Y are both [N, 1] and so is the shape of the output loss.
Since target Y is not differentiable, calculating gradient for Y is illegal.
The formula of modified huber loss is:

$$
L(y, f(x)) = 
\begin{cases}
(\max(0, 1 - yf(x)))^2,  \text{if} \  yf(x) >= -1    \\
             -4yf(x),    \quad \text{otherwise}
\end{cases}
$$

Make sure the values of target label Y are in {0, 1} here. This operator will
scale values of Y to {-1, +1} when computing losses and gradients.

)DOC");
  }
};

class ModifiedHuberLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X must be initialized.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Y must be initialized.");
    PADDLE_ENFORCE(ctx->HasInput("IntermediateVal"),
                   "Intermediate value must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@Grad) must not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto intermediate_dims = ctx->GetInputDim("IntermediateVal");
    auto out_grad_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          intermediate_dims, x_dims,
          "The shape of X and intermediate value must be the same.");
      PADDLE_ENFORCE_EQ(out_grad_dims, x_dims,
                        "The shape of Input(Out@Grad) and X must be the same.");
    }

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(modified_huber_loss, ops::ModifiedHuberLossOp,
                  ops::ModifiedHuberLossOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(modified_huber_loss_grad, ops::ModifiedHuberLossGradOp);

REGISTER_OP_CPU_KERNEL(
    modified_huber_loss,
    ops::ModifiedHuberLossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(modified_huber_loss_grad,
                       ops::ModifiedHuberLossGradCPUKernel<float>);
