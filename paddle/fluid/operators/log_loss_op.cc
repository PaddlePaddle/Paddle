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

#include "paddle/fluid/operators/log_loss_op.h"
#include <memory>

namespace paddle {
namespace operators {

class LogLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Predicted"), "Input", "Predicted", "LogLoss");
    OP_INOUT_CHECK(ctx->HasInput("Labels"), "Input", "Labels", "LogLoss");

    auto pred_dims = ctx->GetInputDim("Predicted");
    auto label_dims = ctx->GetInputDim("Labels");

    if (ctx->IsRuntime() || (framework::product(pred_dims) > 0 &&
                             framework::product(label_dims) > 0)) {
      PADDLE_ENFORCE_EQ(
          pred_dims, label_dims,
          platform::errors::InvalidArgument(
              "The dimensions of Input(Predicted) must be equal to the"
              "dimensions of Input(Labels), but received dimensions of "
              "Input(Predicted)"
              "is [%s], received dimensions of Input(Labels) is [%s].",
              pred_dims, label_dims));
    }
    PADDLE_ENFORCE_EQ(pred_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(Predicted) must be 2,"
                          "But received dimensions of Input(Predicted)"
                          "is [%d]",
                          pred_dims.size()));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          pred_dims[1], 1,
          platform::errors::InvalidArgument(
              "Each row of Input(Predicted) contains a real value, "
              "so the 2nd dimension of Input(X) must be 1,"
              "But got [%d]",
              pred_dims[1]));
    }
    ctx->SetOutputDim("Loss", {pred_dims[0], 1});
    ctx->ShareLoD("Predicted", "Loss");
  }
};

template <typename AttrType>
class LogLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Predicted",
             "The input value (Predicted) of Log loss op."
             "Predicted is a 2-D tensor with shape [batch_size, 1].");
    AddInput("Labels",
             "The target value (Labels) of Log loss op."
             "Labels is a 2-D tensor with shape [batch_size, 1].");
    AddOutput("Loss",
              "The output tensor with shape [batch_size, 1] "
              "which represents the log loss.");
    AddAttr<AttrType>("epsilon", "Epsilon in log loss.");
    AddComment(R"DOC(
LogLoss Operator.

Log loss is a loss function used for binary classification. Log Loss quantifies
the accuracy of a classifier by penalising false classifications. Minimising the
Log Loss is equivalent to maximising the accuracy of the classifier. We define
Predicted as the values predicted by our model and Labels as the target ground
truth value. Log loss can evaluate how close the predicted values are to the
target. The shapes of Predicted and Labels are both [batch_size, 1].
The equation is:

$$
Loss = - Labels * log(Predicted + \epsilon) -
        (1 - Labels) * log(1 - Predicted + \epsilon)
$$

)DOC");
  }
};

class LogLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Predicted"), "Input", "Predicted",
                   "LogLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Labels"), "Input", "Labels", "LogLossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Loss")), "Input",
                   framework::GradVarName("Loss"), "LogLossGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Predicted")),
                   "Output", framework::GradVarName("Predicted"),
                   "LogLossGrad");

    auto pred_dims = ctx->GetInputDim("Predicted");
    auto loss_grad_dims = ctx->GetInputDim(framework::GradVarName("Loss"));
    PADDLE_ENFORCE_EQ(loss_grad_dims, pred_dims,
                      platform::errors::InvalidArgument(
                          "The dimensions of loss_grad must be equal to the "
                          "dimensions of Predicted,"
                          "But received dimensions of loss_grad is [%s], "
                          "received Predicted is "
                          "[%s]",
                          loss_grad_dims, pred_dims));

    auto pred_grad_name = framework::GradVarName("Predicted");
    ctx->SetOutputDim(pred_grad_name, pred_dims);
  }
};

template <typename T>
class LogLossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("log_loss_grad");
    op->SetInput("Predicted", this->Input("Predicted"));
    op->SetInput("Labels", this->Input("Labels"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetOutput(framework::GradVarName("Predicted"),
                  this->InputGrad("Predicted"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(log_loss, ops::LogLossOp, ops::LogLossOpMaker<float>,
                  ops::LogLossGradMaker<paddle::framework::OpDesc>,
                  ops::LogLossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(log_loss_grad, ops::LogLossGradOp);
REGISTER_OP_CPU_KERNEL(
    log_loss, ops::LogLossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    log_loss_grad,
    ops::LogLossGradKernel<paddle::platform::CPUDeviceContext, float>);
