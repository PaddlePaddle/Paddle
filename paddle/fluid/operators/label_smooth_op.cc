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

#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class LabelSmoothOp : public framework::OperatorWithKernel {
 public:
  LabelSmoothOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::NotFound(
                          "The input 'X' of LabelSmoothOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                      true,
                      platform::errors::NotFound(
                          "The output 'Out' of LabelSmoothOp is not found."));
    auto in_dims = ctx->GetInputDim("X");
    if (ctx->HasInput("PriorDist")) {
      auto noise_dims = ctx->GetInputDim("PriorDist");
      auto noise_numel = phi::product(noise_dims);
      PADDLE_ENFORCE_EQ(
          in_dims[in_dims.size() - 1],
          noise_numel,
          platform::errors::InvalidArgument(
              "The number of elements in input 'PriorDist' must be equal to "
              "the "
              "dimension of each label. But received each label's "
              "dimension=[%d], number of elements in input 'PriorDist' is [%d]",
              in_dims[in_dims.size() - 1],
              noise_numel));
    }
    ctx->ShareLoD("X", /*->*/ "Out");
    ctx->SetOutputDim("Out", in_dims);
  }
};

class LabelSmoothOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor) The input labels of LabelSmooth operator. This "
             "input can be batched labels in one-hot encoding or output from "
             "softmax, with shape [N x K], where N is the batch size and K is "
             "the number of classes");
    AddInput("PriorDist",
             "(Tensor, optional)"
             "The prior distribution to be added to the smoothed label. It is "
             "fixed during training and the number of elements should be equal "
             "to the dimension K of each label. Default is uniform "
             "distribution and each element will be set to 1/K if not provided "
             "in input.")
        .AsDispensable();
    AddOutput("Out",
              "(loDTensor) The smoothed label of LabelSmooth operator. It has"
              "the same shape and LoD with the Input(LoDTensor).");
    AddAttr<float>("epsilon",
                   "(float, default 0.0f)"
                   "The smoothing parameter of LabelSmooth operator.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
LabelSmooth Operator.

Label smoothing is a mechanism to regularize the classifier layer. In machine
learning, optimizing the log-likelihood of the correct label directly may
cause two problems. First, it may result in overfitting: if the model learns
to assign full probability to the ground-truth label for each training example,
it is not guaranteed to generalize. Second, it encourages the differences
between the largest logit and all others to become large, reducing the ability
of the model to adapt. Label smoothing is proposed to encourage the model to
be less confident, which replaces the ground-truth label $y$ with the weighted
sum of itself and some fixed distribution $\mu$, i.e.

$$
    \tilde{y} = (1 - \epsilon) * y + \epsilon * \mu,
$$

where $(1 - \epsilon)$ and $\epsilon$ are the weights respectively, and
$\tilde{y}$ is the smoothed label. Usually uniform distribution is used for
$\mu$. This change in the ground-truth label is called label-smoothing
regularization or LSR.

See more details about label smoothing in https://arxiv.org/abs/1512.00567.

)DOC");
  }
};

class LabelSmoothGradOp : public framework::OperatorWithKernel {
 public:
  LabelSmoothGradOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }
};

template <typename T>
class LabelSmoothGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("label_smooth_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(label_smooth,
                  ops::LabelSmoothOp,
                  ops::LabelSmoothOpMaker,
                  ops::LabelSmoothGradMaker<paddle::framework::OpDesc>,
                  ops::LabelSmoothGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(label_smooth_grad, ops::LabelSmoothGradOp);
