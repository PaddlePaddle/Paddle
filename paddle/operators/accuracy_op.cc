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
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Inference"),
                   "Input(Inference) of AccuracyOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input(Label) of AccuracyOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Accuracy"),
                   "Output(Accuracy) of AccuracyOp should not be null.");

    auto inference_dim = ctx->GetInputDim("Inference");
    auto label_dim = ctx->GetInputDim("Label");

    PADDLE_ENFORCE_EQ(label_dim.size(), 1, "label must be a vector");
    PADDLE_ENFORCE_EQ(inference_dim[0], label_dim[0],
                      "inference size must be the same as label size");

    ctx->SetOutputDim("Accuracy", {1});
    ctx->ShareLoD("Inference", /*->*/ "Accuracy");
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

    AddComment(R"DOC(
Accuracy. It will print accuracy rate for classification.
The accuracy is:
..  math::
accuracy = \\frac{NumOfCorrectPredicts}{NumOfAllSamples})

Both the input `Inference` and `Label` can carry the LoD (Level of Details)
information, or not. But the output only shares the LoD with input `Inference`.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(accuracy, ops::AccuracyOp, ops::AccuracyOpMaker);
REGISTER_OP_CPU_KERNEL(accuracy,
                       ops::AccuracyKernel<paddle::platform::CPUPlace, float>);
