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

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

using framework::Tensor;
const int kIgnoreIndex = -100;

class SigmoidCrossEntropyWithLogitsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class SigmoidCrossEntropyWithLogitsGradOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "SigmoidCrossEntropyWithLogitsGradOp");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label",
                   "SigmoidCrossEntropyWithLogitsGradOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"),
                   "SigmoidCrossEntropyWithLogitsGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   framework::GradVarName("X"),
                   "SigmoidCrossEntropyWithLogitsGradOp");

    auto x_dims = ctx->GetInputDim("X");
    auto labels_dims = ctx->GetInputDim("Label");
    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    int rank = x_dims.size();
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(labels_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(
          phi::slice_ddim(x_dims, 0, rank),
          phi::slice_ddim(labels_dims, 0, rank),
          platform::errors::InvalidArgument(
              "Input(X) and Input(Label) shall have the same shape "
              "except the last dimension. But received: the shape of "
              "Input(X) is [%s], the shape of Input(Label) is [%s].",
              x_dims, labels_dims));

      PADDLE_ENFORCE_EQ(
          phi::slice_ddim(x_dims, 0, rank), phi::slice_ddim(dout_dims, 0, rank),
          platform::errors::InvalidArgument(
              "Input(X) and Input(Out@Grad) shall have the same shape "
              "except the last dimension. But received: the shape of "
              "Input(X) is [%s], the shape of Input(Out@Grad) is [%s].",
              x_dims, dout_dims));
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

template <typename T>
class SigmoidCrossEntropyWithLogitsGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sigmoid_cross_entropy_with_logits_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(SigmoidCrossEntropyWithLogitsInplaceInferer,
                           {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(SigmoidCrossEntropyWithLogitsGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(
    sigmoid_cross_entropy_with_logits,
    SigmoidCrossEntropyWithLogitsInferShapeFunctor,
    PD_INFER_META(phi::SigmoidCrossEntropyWithLogitsInferMeta));
REGISTER_OPERATOR(
    sigmoid_cross_entropy_with_logits, ops::SigmoidCrossEntropyWithLogitsOp,
    ops::SigmoidCrossEntropyWithLogitsOpMaker,
    ops::SigmoidCrossEntropyWithLogitsGradOpMaker<paddle::framework::OpDesc>,
    ops::SigmoidCrossEntropyWithLogitsGradOpMaker<paddle::imperative::OpBase>,
    ops::SigmoidCrossEntropyWithLogitsInplaceInferer,
    SigmoidCrossEntropyWithLogitsInferShapeFunctor);
REGISTER_OPERATOR(sigmoid_cross_entropy_with_logits_grad,
                  ops::SigmoidCrossEntropyWithLogitsGradOp,
                  ops::SigmoidCrossEntropyWithLogitsGradInplaceInferer);
