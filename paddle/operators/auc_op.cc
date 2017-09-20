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

class AucOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Inference"),
                            "Input of Inference must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input of Inference must be initialized.");
    auto *inference = ctx.Input<framework::Tensor>("Inference");
    auto *label = ctx.Input<framework::Tensor>("Label");

    PADDLE_ENFORCE_EQ(inference->dims(), label->dims(),
                      "inference and label should have same shape");

    ctx.Output<framework::LoDTensor>("AUC")->Resize({1});
  }
};

class AucOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AucOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Inference",
             "A floating point tensor of arbitrary shape and whose values"
             "are in the range [0, 1].");
    AddInput("Label",
             "A tensor whose shape matches "
             "Inference. Will be cast to bool.");
    // TODO(typhoonzero): support weight input
    AddOutput("AUC",
              "A scalar representing the "
              "current area-under-curve.");

    AddAttr<std::string>("curve", "Curve type, can be 'ROC' or 'PR'.")
        .SetDefault("ROC");
    AddAttr<int>("num_thresholds",
                 "The number of thresholds to use when discretizing the"
                 " roc curve.")
        .SetDefault(200);

    AddComment(
        R"DOC(Computes the AUC according forward output and label.
        Best to use for binary classification evaluations.

        If input label contains values other than 0 and 1, it will be cast
        to bool.

        You can find the definations here: 
        https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
        
        Possible curves are:
        - ROC: Receiver operating characteristic
        - PR: Precision Recall
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(auc, ops::AucOp, ops::AucOpMaker);
REGISTER_OP_CPU_KERNEL(auc, ops::AucKernel<paddle::platform::CPUPlace, float>);
