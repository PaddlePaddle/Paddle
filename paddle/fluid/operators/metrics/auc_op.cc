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

#include "paddle/fluid/operators/metrics/auc_op.h"

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
    PADDLE_INFERSHAPE_ENFORCE_EQ(ctx, predict_width, 2,
                                 "Only support binary classification");
    auto predict_height = ctx->GetInputDim("Predict")[0];
    auto label_height = ctx->GetInputDim("Label")[0];

    PADDLE_INFERSHAPE_ENFORCE_EQ(ctx, predict_height, label_height,
                                 "Out and Label should have same height.");

    int num_pred_buckets = ctx->Attrs().Get<int>("num_thresholds") + 1;
    int slide_steps = ctx->Attrs().Get<int>("slide_steps");

    PADDLE_ENFORCE_GE(num_pred_buckets, 1, "num_thresholds must larger than 1");
    PADDLE_ENFORCE_GE(slide_steps, 0, "slide_steps must be natural number");

    ctx->SetOutputDim("AUC", {1});

    slide_steps = slide_steps == 0 ? 1 : slide_steps;
    ctx->SetOutputDim("StatPosOut", {slide_steps, num_pred_buckets});
    ctx->SetOutputDim("StatNegOut", {slide_steps, num_pred_buckets});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Predict")->type(),
                                   platform::CPUPlace());
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

    // TODO(typhoonzero): support weight input
    AddInput("StatPos", "Statistic value when label = 1");
    AddInput("StatNeg", "Statistic value when label = 0");

    AddOutput("AUC",
              "A scalar representing the "
              "current area-under-the-curve.");

    AddOutput("StatPosOut", "Statistic value when label = 1");
    AddOutput("StatNegOut", "Statistic value when label = 0");

    AddAttr<std::string>("curve", "Curve type, can be 'ROC' or 'PR'.")
        .SetDefault("ROC");

    AddAttr<int>(
        "num_thresholds",
        "The number of thresholds to use when discretizing the roc curve.")
        .SetDefault((2 << 12) - 1);
    AddAttr<int>("slide_steps", "Use slide steps to calc batch auc.")
        .SetDefault(1);
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
