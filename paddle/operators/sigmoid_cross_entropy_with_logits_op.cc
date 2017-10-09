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

#include "paddle/operators/sigmoid_cross_entropy_with_logits_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class SigmoidCrossEntropyWithLogitsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Labels");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2,
                      "Input(Labels)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], labels_dims[0],
                      "The 1st dimension of Input(X) and Input(Labels) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[1], labels_dims[1],
                      "The 2nd dimension of Input(X) and Input(Labels) should "
                      "be equal.");

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class SigmoidCrossEntropyWithLogitsGradOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Labels");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2,
                      "Input(Labels)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(dout_dims.size(), 2,
                      "Input(Out@Grad)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], labels_dims[0],
                      "The 1st dimension of Input(X) and Input(Labels) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[1], labels_dims[1],
                      "The 2nd dimension of Input(X) and Input(Labels) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[0], dout_dims[0],
                      "The 1st dimension of Input(X) and Input(Out@Grad) "
                      "should be equal.");
    PADDLE_ENFORCE_EQ(x_dims[1], dout_dims[1],
                      "The 2nd dimension of Input(X) and Input(Out@Grad) "
                      "should be equal.");

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

class SigmoidCrossEntropyWithLogitsOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  SigmoidCrossEntropyWithLogitsOpMaker(framework::OpProto* proto,
                                       framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D, "
             "where N is the batch size and D is the number of classes. "
             "This input is a tensor of logits computed by the previous "
             " operator. Logits are unscaled log probabilities given as "
             "log(p/(1-p)).");
    AddInput("Labels",
             "(Tensor, default Tensor<float>), a 2-D tensor of the same type "
             "and shape as X. This input is a tensor of probabalistic labels "
             "for each logit");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D "
              " of elementwise logistic losses.");
    AddComment(R"DOC(
SigmoidCrossEntropyWithLogits Operator.

This measures the elementwise probability error in discrete classification tasks
in which each class is independent. This can be thought of as predicting labels
for a data-point that are not mutually exclusive. For example, a news article
can be about politics, technology or sports at the same time or none of these.

The logistic loss is given as follows:

       loss = -Labels * log(sigmoid(X)) - (1 - Labels) * log(1 - sigmoid(X))

We know that sigmoid(X) = (1 / (1 + exp(-X))). By substituting this we get

       loss = X - X * Labels + log(1 + exp(-X))

For stability and to prevent overflow of exp(-X) when X < 0,
we can reformulate the loss as follows:

       loss = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))

Both the input `X` and `Labels` can carry the LoD (Level of Details) information.
However the output only shares the LoD with input `X`.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sigmoid_cross_entropy_with_logits,
            ops::SigmoidCrossEntropyWithLogitsOp,
            ops::SigmoidCrossEntropyWithLogitsOpMaker,
            sigmoid_cross_entropy_with_logits_grad,
            ops::SigmoidCrossEntropyWithLogitsGradOp);
REGISTER_OP_CPU_KERNEL(sigmoid_cross_entropy_with_logits,
                       ops::SigmoidCrossEntropyWithLogitsKernel<
                           paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                       ops::SigmoidCrossEntropyWithLogitsGradKernel<
                           paddle::platform::CPUPlace, float>);
