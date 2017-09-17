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

#include "paddle/operators/accuracy_op.h"

namespace paddle {
namespace operators {

class AccuracyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(
        ctx.InputVar("Inference"),
        "Input(Inference) of AccuracyOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Label) of AccuracyOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(
        ctx.OutputVar("Accuracy"),
        "Output(Accuracy) of AccuracyOp should not be null.");

    auto *inference = ctx.Input<framework::Tensor>("Inference");
    auto *label = ctx.Input<framework::Tensor>("Label");

    PADDLE_ENFORCE_EQ(label->dims().size(), 1, "label must be a vector");
    PADDLE_ENFORCE_EQ(inference->dims()[0], label->dims()[0],
                      "inference size must be the same as label size");

    ctx.Output<framework::LoDTensor>("Accuracy")->Resize({1});
  }
};

class AccuracyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AccuracyOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    // TODO(typhoonzero): support both inference value and indices.
    AddInput("Inference", "topk(indices) the network output");
    AddInput("Label", "Label of the training data");
    // TODO(typhoonzero): AddInput("Weight", ...
    AddOutput("Accuracy", "The accuracy of current batch");

    AddComment(
        R"DOC(Accuracy. It will print accuracy rate for classification.
The accuracy is:
..  math::
accuracy = \\frac{NumOfCorrectPredicts}{NumOfAllSamples})DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(accuracy, ops::AccuracyOp, ops::AccuracyOpMaker);
REGISTER_OP_CPU_KERNEL(accuracy,
                       ops::AccuracyKernel<paddle::platform::CPUPlace, float>);
