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

class SoftmaxWithCrossEntropyOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  SoftmaxWithCrossEntropyOpMaker(framework::OpProto* proto,
                                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Logits",
             "The unscaled log probabilities which is a 2-D tensor<float> with"
             "shape [N x K]. N is the batch_size, and K is the class number.")
        .NotInGradient();
    AddInput("Label", "The ground truth. A 1-D tensor<int> with shape N.");
    AddOutput("Softmax",
              "Store the outputs of softmax function, "
              "which will be used in backward calculation.")
        .AsIntermediate();
    AddOutput("Out", "A 1-D tensor<float> with shape N.");
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
  void InferShape(const framework::InferShapeContext& ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@Grad) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Softmax"),
                            "Input(Softmax) should be not null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Lable) should be not null.");

    ctx.Output<framework::LoDTensor>(framework::GradVarName("Logits"))
        ->Resize(ctx.Input<Tensor>("Softmax")->dims());
  }
};

class SoftmaxWithCrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {
    const Tensor* logits = ctx.Input<Tensor>("Logits");
    PADDLE_ENFORCE(
        logits->dims().size() == 2UL,
        "The input of softmax_with_cross_entropy should be a 2-d tensor.");
    PADDLE_ENFORCE(ctx.Input<Tensor>("Label")->dims().size() == 1UL,
                   "The label should be a 1-d tensor.");

    ctx.Output<framework::LoDTensor>("Softmax")->Resize(logits->dims());
    ctx.Output<framework::LoDTensor>("Out")->Resize({logits->dims()[0], 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyOp,
            ops::SoftmaxWithCrossEntropyOpMaker,
            softmax_with_cross_entropy_grad,
            ops::SoftmaxWithCrossEntropyOpGrad);
REGISTER_OP_CPU_KERNEL(softmax_with_cross_entropy,
                       ops::SoftmaxWithCrossEntropyKernel<float>);
REGISTER_OP_CPU_KERNEL(softmax_with_cross_entropy_grad,
                       ops::SoftmaxWithCrossEntropyGradKernel<float>);
