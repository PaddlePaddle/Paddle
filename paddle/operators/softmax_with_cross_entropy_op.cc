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

#include "paddle/operators/softmax_with_cross_entropy_op.h"

namespace paddle {
namespace operators {

class SoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto logits = ctx.Input<Tensor>("logits");
    PADDLE_ENFORCE(
        logits->dims().size() == 2UL,
        "The input of softmax_with_cross_entropy should be a 2-d tensor.");
    PADDLE_ENFORCE(ctx.Input<Tensor>("lables")->dims().size() == 1UL,
                   "The label should be a 1-d tensor.");
    ctx.Output<Tensor>("Y")->Resize({logits->dims()[0]});
  }
};

class SoftmaxWithCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  SoftmaxWithCrossEntropyOpMaker(framework::OpProto *proto,
                                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("logits",
             "The unscaled log probabilities which is a 2-D tensor<float> with"
             "shape [N x K]. N is the batch_size, and K is the class number.");
    AddInput("label", "The ground truth. A 1-D tensor<int> with shape N.");
    AddOutput("Y", "A 1-D tensor<float> with shape N.");
    AddComment(R"DOC(
Cross entropy loss with softmax are used as the output layer extensively. This
operator computes the softmax normalized values for each row of the input
tensor, after which cross-entropy loss is then computed. This provides a more
numerically stable gradient.

Because this operators performs a softmax on logits internally, it expects
unscaled logits. Please do not call this op with the output of softmax operator,
which will produce incorrect results.

This operators expects mutually exclusive hard labels, each sample in a batch
is in exactly one class with probabilities 1. Each sample in the batch with one
and only one label.
)DOC");
  }
};

class SoftmaxWithCrossEntropyOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Input(Y) should be not null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Y")),
                            "Input(Y@GRAD) should be not null.");
    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("Y")->dims(),
                      ctx.Input<Tensor>(framework::GradVarName("Y"))->dims(),
                      "Input(Y) and its gradients should have a same shape.");

    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("labels"),
                            "Input(lables) should be not null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("logits")),
                            "Input(logits@GRAD) should be not null.");
    PADDLE_ENFORCE_EQ(
        ctx.Input<Tensor>("logits")->dims(),
        ctx.Input<Tensor>(framework::GradVarName("logits"))->dims(),
        "Input(logits) and its gradients should have a same shape.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyOp,
            ops::SoftmaxWithCrossEntropyOpMaker,
            softmax_with_cross_entropy_grad,
            ops::SoftmaxWithCrossEntropyOpGrad);
REGISTER_OP_CPU_KERNEL(
    softmax_with_cross_entropy,
    ops::SoftmaxWithCrossEntropyKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradKernel<paddle::platform::CPUPlace, float>);
