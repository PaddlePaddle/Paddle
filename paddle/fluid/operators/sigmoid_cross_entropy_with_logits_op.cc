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

#include "paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;
const int kIgnoreIndex = -100;

class SigmoidCrossEntropyWithLogitsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");

    int rank = x_dims.size();
    PADDLE_ENFORCE_EQ(rank, labels_dims.size(),
                      "Input(X) and Input(Label) shall have the same rank.");
    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank),
                        framework::slice_ddim(labels_dims, 0, rank),
                        "Input(X) and Input(Label) shall have the same shape "
                        "except the last dimension.");
    }

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class SigmoidCrossEntropyWithLogitsGradOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    int rank = x_dims.size();
    bool check = true;
    if ((!ctx->IsRuntime()) && (framework::product(x_dims) <= 0 ||
                                framework::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 0, rank),
                        framework::slice_ddim(labels_dims, 0, rank),
                        "Input(X) and Input(Label) shall have the same shape.");

      PADDLE_ENFORCE_EQ(
          framework::slice_ddim(x_dims, 0, rank),
          framework::slice_ddim(dout_dims, 0, rank),
          "Input(X) and Input(Out@Grad) shall have the same shape.");
    }

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

class SigmoidCrossEntropyWithLogitsOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D, "
             "where N is the batch size and D is the number of classes. "
             "This input is a tensor of logits computed by the previous "
             " operator. Logits are unscaled log probabilities given as "
             "log(p/(1-p)).");
    AddInput("Label",
             "(Tensor, default Tensor<float>), a 2-D tensor of the same type "
             "and shape as X. This input is a tensor of probabalistic labels "
             "for each logit");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D "
              " of elementwise logistic losses.");
    AddAttr<bool>("normalize",
                  "if true, divide the loss by the number of "
                  "targets != ignore_index.")
        .SetDefault(false);
    AddAttr<int>("ignore_index",
                 "(int, default kIgnoreIndex), Specifies a target value that "
                 "is ignored and"
                 "does not contribute to the input gradient.")
        .SetDefault(kIgnoreIndex);
    AddComment(R"DOC(
SigmoidCrossEntropyWithLogits Operator.

This measures the element-wise probability error in classification tasks
in which each class is independent. This can be thought of as predicting labels
for a data-point, where labels are not mutually exclusive.
For example, a news article can be about politics, technology or sports
at the same time or none of these.

The logistic loss is given as follows:

       $$loss = -Labels * \log(\sigma(X)) - (1 - Labels) * \log(1 - \sigma(X))$$

We know that $$\sigma(X) = \\frac{1}{1 + \exp(-X)}$$. By substituting this we get:

       $$loss = X - X * Labels + \log(1 + \exp(-X))$$

For stability and to prevent overflow of $$\exp(-X)$$ when X < 0,
we reformulate the loss as follows:

       $$loss = \max(X, 0) - X * Labels + \log(1 + \exp(-\|X\|))$$

Both the input `X` and `Labels` can carry the LoD (Level of Details) information.
However the output only shares the LoD with input `X`.

)DOC");
  }
};

class SigmoidCrossEntropyWithLogitsGradOpDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("sigmoid_cross_entropy_with_logits_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Label", Input("Label"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sigmoid_cross_entropy_with_logits,
                  ops::SigmoidCrossEntropyWithLogitsOp,
                  ops::SigmoidCrossEntropyWithLogitsOpMaker,
                  ops::SigmoidCrossEntropyWithLogitsGradOpDescMaker);
REGISTER_OPERATOR(sigmoid_cross_entropy_with_logits_grad,
                  ops::SigmoidCrossEntropyWithLogitsGradOp);
REGISTER_OP_CPU_KERNEL(
    sigmoid_cross_entropy_with_logits,
    ops::SigmoidCrossEntropyWithLogitsKernel<paddle::platform::CPUDeviceContext,
                                             float>,
    ops::SigmoidCrossEntropyWithLogitsKernel<paddle::platform::CPUDeviceContext,
                                             double>);
REGISTER_OP_CPU_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                       ops::SigmoidCrossEntropyWithLogitsGradKernel<
                           paddle::platform::CPUDeviceContext, float>,
                       ops::SigmoidCrossEntropyWithLogitsGradKernel<
                           paddle::platform::CPUDeviceContext, double>);
