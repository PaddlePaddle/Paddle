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

#include "paddle/fluid/operators/rank_loss_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class RankLossOp : public framework::OperatorWithKernel {
 public:
  RankLossOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Label"), true,
                      "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Left"), true,
                      "Input(Left) shouldn't be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Right"), true,
                      "Input(Right) shouldn't be null.");

    auto label_dims = ctx->GetInputDim("Label");
    auto left_dims = ctx->GetInputDim("Left");
    auto right_dims = ctx->GetInputDim("Right");
    // check label_dims valid
    PADDLE_ENFORCE_GE(label_dims.size(), 1,
                      "The dimension size of Input(Label) must be greater than "
                      "or equal to 1.");
    PADDLE_ENFORCE_LE(
        label_dims.size(), 2,
        "The dimension size of Input(Label) must be less than or equal to 2.");
    if (label_dims.size() == 2U) {
      PADDLE_ENFORCE_EQ(label_dims[1], 1,
                        "The last dimension of Input(Label) must be 1.");
    }
    // check left_dims valid
    PADDLE_ENFORCE_GE(left_dims.size(), 1,
                      "The dimension size of Input(Left) must be greater than "
                      "or equal to 1.");
    PADDLE_ENFORCE_LE(
        left_dims.size(), 2,
        "The dimension size of Input(Left) must be less than or equal to 2.");
    if (left_dims.size() == 2U) {
      PADDLE_ENFORCE_EQ(left_dims[1], 1,
                        "The last dimension of Input(Left) must be 1.");
    }
    // check right_dims valid
    PADDLE_ENFORCE_GE(right_dims.size(), 1,
                      "The dimension size of Input(Right) must be greater than "
                      "or equal to 1.");
    PADDLE_ENFORCE_LE(
        right_dims.size(), 2,
        "The dimension size of Input(Right) must be less than or equal to 2.");
    if (right_dims.size() == 2U) {
      PADDLE_ENFORCE_EQ(right_dims[1], 1,
                        "The last dimension of Input(Right) must be 1.");
    }
    PADDLE_ENFORCE_EQ(label_dims[0], left_dims[0],
                      "The first dimension of Input(Label) and Input(Left) "
                      "must have the same value.");
    PADDLE_ENFORCE_EQ(label_dims[0], right_dims[0],
                      "The first dimension of Input(Label) and Input(Right) "
                      "must have the same value.");
    ctx->SetOutputDim("Out", label_dims);
  }
};

class RankLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Label",
             "(2-D Tensor with shape [batch_size x 1]) "
             "The label indicating A ranked higher than B or not.");
    AddInput("Left",
             "(2-D Tensor with shape [batch_size x 1]) "
             "The output of RankNet for doc A.");
    AddInput("Right",
             "(2-D Tensor with shape [batch_size x 1]) "
             "The output of RankNet for doc B.");
    AddOutput("Out",
              "(2-D Tensor with shape [batch_size x 1]) "
              "The output loss of RankLoss operator.");
    AddComment(R"DOC(
RankLoss Operator.

RankLoss operator for RankNet
(http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf). 
RankNet is a pairwise ranking model with
one training sample consisting of a pair of doc A and B, and the label P
indicating that A is ranked higher than B or not:

P = {0, 1} or {0, 0.5, 1}, where 0.5 means no information about the rank of
the input pair.

The RankLoss operator takes three inputs: Left (o_i), Right (o_j) and Label
(P_{i,j}), which represent the output score of RankNet for the two docs and 
the label respectively, and yields the rank loss C_{i,j} using the following 
equation:

$$
  C_{i,j} = -\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\
  o_{i,j} =  o_i - o_j  \\
  \tilde{P_{i,j}} = \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}
$$

The operator can take batch inputs with size batch_size (batch_size >= 1).

)DOC");
  }
};

class RankLossGradOp : public framework::OperatorWithKernel {
 public:
  RankLossGradOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Label"), true,
                      "Input(Label) shouldn't be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Left"), true,
                      "Input(Left) shouldn't be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Right"), true,
                      "Input(Right) shouldn't be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      "Input(Out@GRAD) shouldn't be null.");
    auto left_dims = ctx->GetInputDim("Left");
    auto right_dims = ctx->GetInputDim("Right");
    auto left_grad_name = framework::GradVarName("Left");
    auto right_grad_name = framework::GradVarName("Right");

    if (ctx->HasOutput(left_grad_name)) {
      ctx->SetOutputDim(left_grad_name, left_dims);
    }

    if (ctx->HasOutput(right_grad_name)) {
      ctx->SetOutputDim(right_grad_name, right_dims);
    }
  }
};

class RankLossGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("rank_loss_grad");
    op->SetInput("Label", Input("Label"));
    op->SetInput("Left", Input("Left"));
    op->SetInput("Right", Input("Right"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Left"), InputGrad("Left"));
    op->SetOutput(framework::GradVarName("Right"), InputGrad("Right"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(rank_loss, ops::RankLossOp, ops::RankLossOpMaker,
                  ops::RankLossGradDescMaker);
REGISTER_OPERATOR(rank_loss_grad, ops::RankLossGradOp);
REGISTER_OP_CPU_KERNEL(
    rank_loss, ops::RankLossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    rank_loss_grad,
    ops::RankLossGradKernel<paddle::platform::CPUDeviceContext, float>);
