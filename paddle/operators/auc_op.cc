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

#include "paddle/operators/auc_op.h"

namespace paddle {
namespace operators {

class AccuracyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Inference"),
                            "Input of Inference must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input of Inference must be initialized.");
    auto *inference = ctx.Input<framework::Tensor>("Inference");
    auto *inference_prob = ctx.Input<framework::Tensor>("InferenceProb");
    auto *label = ctx.Input<framework::Tensor>("Label");

    PADDLE_ENFORCE_EQ(label->dims().size(), 1, "label must be a vector");
    PADDLE_ENFORCE_EQ(inference->dims()[0], label->dims()[0],
                      "inference size must be the same as label size");
    PADDLE_ENFORCE_EQ(inference->dims(), inference_prob->dims());

    ctx.Output<Tensor>("Accuracy")->Resize({1});
  }
};

class AucOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AucOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Inference",
             "Topk(indices) the network output, float value indicating "
             "probabilities of classification");
    AddInput("InferenceProb",
             "Topk(values) the network output, float value indicating "
             "probabilities of classification");
    AddInput("Label", "Label of the training data");
    // TODO(typhoonzero): support weight
    AddOutput("AUC", "Area Under Curve caculations");
    AddAttr<std::string>("curve", "Possible curves are ROC and PR")
        .SetDefault("ROC");
    AddAttr<int>("num_thresholds",
                 "The number of thresholds to use when discretizing the"
                 " roc curve.")
        .SetDefault(200);

    AddComment(
        R"DOC(Computes the AUC according forward output and label.
        You can find the definations here: 
        https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
        
        Possible curves are:
        ROC: Receiver operating characteristic
        PR: Precision Recall
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(auc, ops::AccuracyOp, ops::AccuracyOpMaker);
REGISTER_OP_CPU_KERNEL(auc, ops::AucKernel<paddle::platform::CPUPlace, float>);
