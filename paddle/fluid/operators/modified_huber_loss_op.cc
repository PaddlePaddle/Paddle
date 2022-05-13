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
#include <memory>

namespace paddle {
namespace operators {

class ModifiedHuberLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ModifiedHuberLoss");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ModifiedHuberLoss");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::InvalidArgument(
                                            "Input(input) rank should be 2, "
                                            "but received input rank(%d) != 2",
                                            x_dims.size()));

    if (ctx->IsRuntime() ||
        (phi::product(x_dims) > 0 && phi::product(y_dims) > 0)) {
      PADDLE_ENFORCE_EQ(
          x_dims, y_dims,
          platform::errors::InvalidArgument(
              "The Input(input) and Input(label) should have the same "
              "shape, but received input shape [%s] != label shape [%s]",
              x_dims, y_dims));
    }

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The second dimension of Input(input) should be 1, "
                            "but received second dimension of input (%d) != 1",
                            x_dims[1]));
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
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ModifiedHuberLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("IntermediateVal"), "Input", "IntermediateVal",
                   "ModifiedHuberLossGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "ModifiedHuberLossGrad");

    auto y_dims = ctx->GetInputDim("Y");
    auto intermediate_dims = ctx->GetInputDim("IntermediateVal");
    auto out_grad_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          intermediate_dims, y_dims,
          platform::errors::InvalidArgument(
              "The shape of Intermediate variable which will be reused in "
              "backward processing should the same as "
              "the shape of Input(label), but received Intermediate variable "
              "shape [%s] != label shape [%s]",
              intermediate_dims, y_dims));

      PADDLE_ENFORCE_EQ(
          out_grad_dims, y_dims,
          platform::errors::InvalidArgument(
              "The shape of output gradient should be the same as "
              "the shape of Input(label), but received the output gradient "
              "shape [%s] != label shape [%s]",
              out_grad_dims, y_dims));
    }

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), y_dims);
    }
  }
};

template <typename T>
class ModifiedHuberLossGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("modified_huber_loss_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("IntermediateVal", this->Output("IntermediateVal"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    modified_huber_loss, ops::ModifiedHuberLossOp,
    ops::ModifiedHuberLossOpMaker,
    ops::ModifiedHuberLossGradOpMaker<paddle::framework::OpDesc>,
    ops::ModifiedHuberLossGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(modified_huber_loss_grad, ops::ModifiedHuberLossGradOp);

REGISTER_OP_CPU_KERNEL(
    modified_huber_loss,
    ops::ModifiedHuberLossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(modified_huber_loss_grad,
                       ops::ModifiedHuberLossGradCPUKernel<float>);
