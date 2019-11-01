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

#include "paddle/fluid/operators/margin_rank_loss_op.h"
#include <memory>

namespace paddle {
namespace operators {

class MarginRankLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // input check
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("X1"), "Input(X1) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("X2"), "Input(X2) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) shouldn't be null.");
    auto label_dims = ctx->GetInputDim("Label");
    auto x1_dims = ctx->GetInputDim("X1");
    auto x2_dims = ctx->GetInputDim("X2");
    PADDLE_ENFORCE(
        (label_dims == x1_dims) && (x1_dims == x2_dims) &&
            (label_dims.size() == 2) && (label_dims[1] == 1),
        "All inputs must be 2-D tensor with shape [batch_size x 1].");
    ctx->SetOutputDim("Activated", label_dims);
    ctx->SetOutputDim("Out", label_dims);
  }
};

template <typename T>
class MarginRankLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X1",
             "(2-D tensor with shape [batch_size x 1]) The score for "
             "one item X1 to be ranked, from pairwise ranking model.");
    AddInput("X2",
             "(2-D tensor with shape [batch_size x 1]) The score for "
             "another item X2 to be ranked, from pairwise ranking model.");
    AddInput("Label",
             "(2-D tensor with shape [batch_size x 1]) "
             "The label indicating X1 ranked higher than X2 or not, "
             "can only be +1 or -1.");
    AddOutput("Activated",
              "(2-D tensor with shape [batch_size x 1]) Intermediate tensor "
              "to indicate whether each element of Output(Out) is activated.")
        .AsIntermediate();
    AddOutput("Out",
              "(2-D tensor with shape [batch_size x 1]) "
              "The output loss of MarginRankLoss operator.");
    AddAttr<T>("margin", "(scalar, default 0) Margin for MarginRankLossOp.")
        .SetDefault(static_cast<T>(0));
    AddComment(R"DOC(
MarginRankLoss Operator.

This operator measures the loss given a pair of training sample
{`X1`, `X2`} and the `Label` with attribute `margin`, where `Label = +1` 
indicating X1 is ranked higher than `X2` and `Label = -1` otherwise. The loss 
is calculated as:

$loss(X1, X2, Label) = \max(0, -Label * (X1 - X2) + margin)$

The attribute `margin` here helps make the predictions more robust.
Denote the item ranked higher as the positive sample, otherwise the negative 
sample. If the score of the two samples satisfies 

$positive sample - negative sample < margin$

the pair of samples will contribute to the final loss, which will backpropagate 
and train the ranking model to enlarge the difference between the two scores.

For batch input with size `batch_size`, `X1`, `X2` and `Label`
all have the same shape [batch_size x 1].

)DOC");
  }
};

class MarginRankLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput("Activated"),
                   "Intermediate(Activated) shouldn't be null.");
    auto dims = ctx->GetInputDim("Label");
    ctx->SetOutputDim(framework::GradVarName("X1"), dims);
    ctx->SetOutputDim(framework::GradVarName("X2"), dims);
  }
};

template <typename T>
class MarginRankLossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("margin_rank_loss_grad");
    op->SetInput("Activated", this->Output("Activated"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Label", this->Input("Label"));
    op->SetOutput(framework::GradVarName("X1"), this->InputGrad("X1"));
    op->SetOutput(framework::GradVarName("X2"), this->InputGrad("X2"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(margin_rank_loss, ops::MarginRankLossOp,
                  ops::MarginRankLossOpMaker<float>,
                  ops::MarginRankLossGradMaker<paddle::framework::OpDesc>,
                  ops::MarginRankLossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(margin_rank_loss_grad, ops::MarginRankLossGradOp);
REGISTER_OP_CPU_KERNEL(
    margin_rank_loss,
    ops::MarginRankLossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    margin_rank_loss_grad,
    ops::MarginRankLossGradKernel<paddle::platform::CPUDeviceContext, float>);
