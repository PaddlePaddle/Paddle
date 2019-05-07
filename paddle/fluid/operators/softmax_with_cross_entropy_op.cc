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

#include "paddle/fluid/operators/softmax_with_cross_entropy_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {

class SoftmaxWithCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The input tensor of unscaled "
             "log probabilities, whose dimension :attr:`axis` should be scaled "
             "by softmax.");
    AddInput(
        "Label",
        "(Tensor) The input tesnor of groud truth label. If :attr:`soft_label` "
        "is set to false, Label is a Tensor<int64> in same shape with "
        "Input(Logits) except the shape in dimension :attr:`axis` as 1. If "
        "soft_label is set to true, Label is a Tensor<float/double> in same "
        "shape with Input(Logits).");
    AddOutput(
        "Softmax",
        "(Tensor, default: Tensor<float>), A tensor in same shape with "
        "Input(Logits). "
        "The outputs value of softmax activation by given the input batch, "
        "which will be used in backward calculation.")
        .AsIntermediate();
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A tensor in same shape with "
              "Input(Logits) "
              "except the shape in dimension :attr:`axis` as 1. The cross "
              "entropy loss.");
    AddAttr<bool>(
        "soft_label",
        "(bool, default: false), A flag to indicate whether to interpretate "
        "the given labels as soft labels.")
        .SetDefault(false);
    AddAttr<bool>(
        "numeric_stable_mode",
        "(bool, default: true), A flag to indicate whether to use more "
        "numerically stable algorithm. This flag is only valid when "
        "soft_label is false and GPU is used.")
        .SetDefault(true);
    AddAttr<int>(
        "ignore_index",
        "(int, default -100), Specifies a target value that is ignored and"
        "does not contribute to the input gradient. Only valid if soft_label"
        "is set to False")
        .SetDefault(-100);
    AddAttr<int>("axis",
                 "The dimension index of Input(Logits) to perform softmax,"
                 "default -1 for last dimension")
        .SetDefault(-1);
    AddComment(R"DOC(
Softmax With Cross Entropy Operator.

Cross entropy loss with softmax is used as the output layer extensively. This
operator computes the softmax normalized values for each row of the input
tensor, after which cross-entropy loss is computed. This provides a more
numerically stable gradient.

Because this operator performs a softmax on logits internally, it expects
unscaled logits. This operator should not be used with the output of
softmax operator since that would produce incorrect results.

When the attribute soft_label is set false, this operators expects mutually
exclusive hard labels, each sample in a batch is in exactly one class with a
probability of 1.0. Each sample in the batch will have a single label.

The equation is as follows:

1) Hard label (one-hot label, so every sample has exactly one class)

$$Loss_j =  -\text{Logit}_{Label_j} +
\log\left(\sum_{i=0}^{K}\exp(\text{Logit}_i)\right),
j = 1,..., K$$

2) Soft label (each sample can have a distribution over all classes)

$$Loss_j =  -\sum_{i=0}^{K}\text{Label}_i \left(\text{Logit}_i -
\log\left(\sum_{i=0}^{K}\exp(\text{Logit}_i)\right)\right),
j = 1,...,K$$

)DOC");
  }
};

class SoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Softmax"),
                   "Output(Softmax) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"), "Output(Loss) should be not null.");

    auto axis = ctx->Attrs().Get<int>("axis");
    auto logits_dims = ctx->GetInputDim("Logits");
    auto labels_dims = ctx->GetInputDim("Label");
    auto logits_rank = logits_dims.size();
    PADDLE_ENFORCE(axis >= -logits_rank && axis < logits_rank,
                   "Attr(axis) value should be in range [-R, R-1], "
                   "R is the rank of Input(Logits).");

    axis = CanonicalAxis(axis, logits_rank);
    for (int i = 0; i < logits_rank; i++) {
      if (i != axis) {
        if (ctx->IsRuntime() || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
          PADDLE_ENFORCE_EQ(
              logits_dims[i], labels_dims[i],
              "Input(Logits) and Input(Label) should in same shape in "
              "dimensions except axis.");
        }
      }
    }

    auto numeric_stable_mode = ctx->Attrs().Get<bool>("numeric_stable_mode");
    if (axis != logits_rank - 1) {
      PADDLE_ENFORCE(
          numeric_stable_mode,
          "Attr(axis) can only be -1 when not in numeric_stable_mode.");
    }

    bool soft_label = ctx->Attrs().Get<bool>("soft_label");
    if (soft_label) {
      if (ctx->IsRuntime() ||
          (logits_dims[axis] > 0 && labels_dims[axis] > 0)) {
        PADDLE_ENFORCE_EQ(logits_dims[axis], labels_dims[axis],
                          "If Attr(soft_label) == true, the axis dimension of "
                          "Input(X) and Input(Label) should be equal.");
      }
    } else {
      if (ctx->IsRuntime() || labels_dims[axis] > 0) {
        PADDLE_ENFORCE_EQ(labels_dims[axis], 1UL,
                          "If Attr(soft_label) == false, the axis dimension of "
                          "Input(Label) should be 1.");
      }
    }

    ctx->SetOutputDim("Softmax", logits_dims);

    logits_dims[axis] = 1;
    ctx->SetOutputDim("Loss", logits_dims);

    ctx->ShareLoD("Logits", /*->*/ "Softmax");
    ctx->ShareLoD("Logits", /*->*/ "Loss");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Logits")->type(),
                                   ctx.device_context());
  }
};

class SoftmaxWithCrossEntropyOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@Grad) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Softmax"),
                   "Input(Softmax) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@Grad) should be not null.");

    auto axis = ctx->Attrs().Get<int>("axis");
    auto softmax_dims = ctx->GetInputDim("Softmax");
    auto labels_dims = ctx->GetInputDim("Label");
    auto softmax_rank = softmax_dims.size();
    PADDLE_ENFORCE(axis >= -softmax_rank && axis < softmax_rank,
                   "Attr(axis) value should be in range [-R, R-1], "
                   "R is the rank of Input(Logits).");

    axis = CanonicalAxis(axis, softmax_rank);
    for (int i = 0; i < softmax_rank; i++) {
      if (i != axis) {
        if (ctx->IsRuntime() || (softmax_dims[i] > 0 && labels_dims[i] > 0)) {
          PADDLE_ENFORCE_EQ(
              softmax_dims[i], labels_dims[i],
              "Input(Logits) and Input(Label) should in same shape in "
              "dimensions except axis.");
        }
      }
    }

    bool soft_label = ctx->Attrs().Get<bool>("soft_label");
    if (soft_label) {
      if (ctx->IsRuntime() ||
          (softmax_dims[axis] > 0 && labels_dims[axis] > 0)) {
        PADDLE_ENFORCE_EQ(softmax_dims[axis], labels_dims[axis],
                          "If Attr(soft_label) == true, the axis dimension of "
                          "Input(X) and Input(Label) should be equal.");
      }
    } else {
      if (ctx->IsRuntime() || labels_dims[axis] > 0) {
        PADDLE_ENFORCE_EQ(labels_dims[axis], 1UL,
                          "If Attr(soft_label) == false, the axis dimension of "
                          "Input(Label) should be 1.");
      }
    }

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Softmax"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Loss"))->type(),
        ctx.device_context());
  }
};

class SoftmaxGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* grad_op = new framework::OpDesc();
    grad_op->SetType("softmax_with_cross_entropy_grad");
    grad_op->SetInput("Label", Input("Label"));
    grad_op->SetInput("Softmax", Output("Softmax"));
    grad_op->SetInput(framework::GradVarName("Softmax"), OutputGrad("Softmax"));
    grad_op->SetInput(framework::GradVarName("Loss"), OutputGrad("Loss"));
    grad_op->SetOutput(framework::GradVarName("Logits"), InputGrad("Logits"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class SoftmaxWithCrossEntropyInplaceInference
    : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc& op_desc, bool use_cuda) const {
    if (use_cuda && !boost::get<bool>(op_desc.GetAttr("soft_label"))) {
      return {{"Logits", "Softmax"}};
    } else {
      return {};
    }
  }
};

class SoftmaxWithCrossEntropyGradInplaceInference
    : public framework::InplaceOpInference {
 public:
  std::unordered_map<std::string, std::string> operator()(
      const framework::OpDesc& op_desc, bool use_cuda) const {
    return {{"Softmax", framework::GradVarName("Logits")}};
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyOp,
                  ops::SoftmaxWithCrossEntropyOpMaker, ops::SoftmaxGradMaker,
                  ops::SoftmaxWithCrossEntropyInplaceInference);
REGISTER_OPERATOR(softmax_with_cross_entropy_grad,
                  ops::SoftmaxWithCrossEntropyOpGrad,
                  ops::SoftmaxWithCrossEntropyGradInplaceInference);
REGISTER_OP_CPU_KERNEL(softmax_with_cross_entropy,
                       ops::SoftmaxWithCrossEntropyKernel<float>,
                       ops::SoftmaxWithCrossEntropyKernel<double>);
REGISTER_OP_CPU_KERNEL(softmax_with_cross_entropy_grad,
                       ops::SoftmaxWithCrossEntropyGradKernel<float>,
                       ops::SoftmaxWithCrossEntropyGradKernel<double>);
