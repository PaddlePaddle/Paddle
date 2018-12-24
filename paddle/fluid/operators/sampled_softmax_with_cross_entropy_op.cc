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

#include "paddle/fluid/operators/sampled_softmax_with_cross_entropy_op.h"
#include "paddle/fluid/operators/math/sample_prob.h"

namespace paddle {
namespace operators {

class SampledSoftmaxWithCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The unscaled log probabilities "
             "which is a 2-D tensor with shape [N x K]. N is the batch_size, "
             "and K is the class number.");
    AddInput("Label",
             "(Tensor) The ground truth which is a 2-D tensor. Label is a "
             "Tensor<int64> with shape [N x NT], where NT is the number of"
             "true labels for each example.");
    AddInput(
        "CustomSamples",
        "(Tensor, default: Tensor<int64_t>), A 2-D tensor with shaoe [N x "
        "S+NT]."
        "The customized sample labels with true labels at first. This tensor"
        "is only use_custom_samples is true.");
    AddInput(
        "CustomProbabilities",
        "(Tensor, default: Tensor<float>), A 2-D tensor with shaoe [N x S+NT]."
        "The customized sample probabilities with true labels at first. This "
        "tensor is only use_custom_samples is true.");

    AddOutput(
        "Samples",
        "(Tensor, default: Tensor<int64_t>), A 2-D tensor with shape [N x "
        "S+NT]."
        "The outputs value of sampler by given the true label, where S is the "
        "number of negative sample for each example. So Samples includes NT "
        "true"
        "labels and S negative labels for each example. This will be used in"
        "backward calculation.")
        .AsIntermediate();
    AddOutput("SampledSoftmax",
              "(Tensor, default: Tensor<float>), A 2-D tensor with shape"
              "[N x S+NT]. The outputs value of sampled softmax, which will be"
              "used in backward calculation.")
        .AsIntermediate();
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A 2-D tensor. The cross "
              "entropy loss with shape [N x 1].");

    AddAttr<bool>(
        "use_custom_samples",
        "An indicator whether to use custom samples with probabilities, if True"
        "the operator will use custom samples and custom probabilities"
        "otherwise, the operator will generate them by itself.")
        .SetDefault(false);
    AddAttr<int>("num_samples", "The number of negative samples.");
    AddAttr<int>("seed", "Random seed for generating samples").SetDefault(0);

    AddComment(R"DOC(
TODO(chenfeiyu): Write documentation for this Operator.
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

class SampledSoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");

    PADDLE_ENFORCE(ctx->HasOutput("Samples"),
                   "Output(Samples) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("SampledSoftmax"),
                   "Output(SampledSoftmax) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"), "Output(Loss) should be not null.");

    auto logits_dims = ctx->GetInputDim("Logits");
    auto labels_dims = ctx->GetInputDim("Label");

    PADDLE_ENFORCE_EQ(
        logits_dims.size(), 2UL,
        "The logits of softmax_with_cross_entropy should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2UL,
                      "The labels should be a 2-D tensor.");

    const int num_samples = ctx->Attrs().Get<int>("num_samples");
    const int num_sampled_classes = labels_dims[1] + num_samples;
    ctx->SetOutputDim("Samples", {logits_dims[0], num_sampled_classes});
    ctx->SetOutputDim("SampledSoftmax", {logits_dims[0], num_sampled_classes});
    ctx->SetOutputDim("Loss", {logits_dims[0], 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Logits"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

// UNDERSTAND: InferShape for Grad
class SampledSoftmaxWithCrossEntropyOpGrad
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Samples"),
                   "Input(Samples) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("SampledSoftmax"),
                   "Input(SampledSoftmax) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@Grad) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@Grad) should be not null.");

    auto logit_dims = ctx->GetInputDim("Logits");
    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2UL,
                      "The label should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(logit_dims.size(), 2UL,
                      "The logits should be a 2-D tensor.");

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Logits"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("Loss")));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

// UNDERSTAND: what's the rule for making a GradMaker TODO
class SampledSoftmaxGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    /* UNDERSTAND: new a OpDesc, then, what SetInput(), SetOutput(), SetType()
    does?
    */
    auto* grad_op = new framework::OpDesc();
    grad_op->SetType("sampled_softmax_with_cross_entropy_grad");
    grad_op->SetInput("Logits", Input("Logits"));
    grad_op->SetInput("Label", Input("Label"));
    grad_op->SetInput("Samples", Output("Samples"));
    grad_op->SetInput("SampledSoftmax", Output("SampledSoftmax"));
    grad_op->SetInput(framework::GradVarName("Loss"), OutputGrad("Loss"));
    grad_op->SetOutput(framework::GradVarName("Logits"), InputGrad("Logits"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sampled_softmax_with_cross_entropy,
                  ops::SampledSoftmaxWithCrossEntropyOp,
                  ops::SampledSoftmaxWithCrossEntropyOpMaker,
                  ops::SampledSoftmaxGradMaker);
REGISTER_OPERATOR(sampled_softmax_with_cross_entropy_grad,
                  ops::SampledSoftmaxWithCrossEntropyOpGrad);
REGISTER_OP_CPU_KERNEL(sampled_softmax_with_cross_entropy,
                       ops::SampledSoftmaxWithCrossEntropyKernel<float>,
                       ops::SampledSoftmaxWithCrossEntropyKernel<double>);
REGISTER_OP_CPU_KERNEL(sampled_softmax_with_cross_entropy_grad,
                       ops::SampledSoftmaxWithCrossEntropyGradKernel<float>,
                       ops::SampledSoftmaxWithCrossEntropyGradKernel<double>);
