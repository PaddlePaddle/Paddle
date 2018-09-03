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

#include "paddle/fluid/operators/auc_op.h"
#include <string>

namespace paddle {
namespace operators {

class AucOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Predict"),
                   "Input of Out should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input of Label should not be null.");
    auto predict_width = ctx->GetInputDim("Predict")[1];
    PADDLE_ENFORCE_EQ(predict_width, 2, "Only support binary classification");
    auto predict_height = ctx->GetInputDim("Predict")[0];
    auto label_height = ctx->GetInputDim("Label")[0];

    PADDLE_ENFORCE_EQ(predict_height, label_height,
                      "Out and Label should have same height.");

    int num_thres = ctx->Attrs().Get<int>("num_thresholds");

    ctx->SetOutputDim("AUC", {1});
    ctx->SetOutputDim("TPOut", {num_thres});
    ctx->SetOutputDim("TNOut", {num_thres});
    ctx->SetOutputDim("FPOut", {num_thres});
    ctx->SetOutputDim("FNOut", {num_thres});

    ctx->ShareLoD("Predict", /*->*/ "AUC");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Predict")->type()),
        ctx.device_context());
  }
};

class AucOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Predict",
             "A floating point 2D tensor with shape [batch_size, 2], values "
             "are in the range [0, 1]."
             "Typically, this tensor indicates the probability of each label");
    AddInput("Label",
             "A 2D int tensor indicating the label of the training data. "
             "shape: [batch_size, 1]");
    AddInput("TP", "True-Positive value.");
    AddInput("FP", "False-Positive value.");
    AddInput("TN", "True-Negative value.");
    AddInput("FN", "False-Negative value.");
    // TODO(typhoonzero): support weight input
    AddOutput("AUC",
              "A scalar representing the "
              "current area-under-the-curve.");
    AddOutput("TPOut", "True-Positive value.");
    AddOutput("FPOut", "False-Positive value.");
    AddOutput("TNOut", "True-Negative value.");
    AddOutput("FNOut", "False-Negative value.");

    AddAttr<std::string>("curve", "Curve type, can be 'ROC' or 'PR'.")
        .SetDefault("ROC");
    AddAttr<int>("num_thresholds",
                 "The number of thresholds to use when discretizing the"
                 " roc curve.")
        .SetDefault(200);

    AddComment(R"DOC(
Area Under The Curve (AUC) Operator.

This implementation computes the AUC according to forward output and label.
It is used very widely in binary classification evaluation. As a note:
If input label contains values other than 0 and 1, it will be cast
to bool. You can find the relevant definitions here:
https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

There are two types of possible curves:
1. ROC: Receiver operating characteristic
2. PR: Precision Recall
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(auc, ops::AucOp, ops::AucOpMaker);
REGISTER_OP_CPU_KERNEL(auc, ops::AucKernel<paddle::platform::CPUPlace, float>);
