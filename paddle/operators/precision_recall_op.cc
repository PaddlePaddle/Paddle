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

#include "paddle/operators/precision_recall_op.h"

namespace paddle {
namespace operators {

class PrecisionRecallOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Predictions"),
                   "Input(Predictions) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Labels"),
                   "Input(Labels) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("BatchMetrics"),
                   "Output(BatchMetrics) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("AccumMetrics"),
                   "Output(AccumMetrics) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("AccumStatesInfo"),
                   "Output(AccumStatesInfo) should not be null.");

    auto predictions_dims = ctx->GetInputDim("Predictions");
    auto labels_dims = ctx->GetInputDim("Labels");

    if (ctx->HasInput("Weights")) {
      auto weights_dims = ctx->GetInputDim("Weights");
      PADDLE_ENFORCE_EQ(weights_dims,
                        framework::make_ddim({predictions_dims[0], 1}),
                        "The shape of Input(Weights) should be "
                        "[batch_size, 1].");
    }
    if (ctx->HasInput("StatesInfo")) {
      auto states_dims = ctx->GetInputDim("StatesInfo");
      PADDLE_ENFORCE_EQ(states_dims,
                        framework::make_ddim({predictions_dims[1], 4}),
                        "The shape of Input(StatesInfo) should be "
                        "[class_number, 4].");
    }
    PADDLE_ENFORCE_EQ(predictions_dims[0], labels_dims[0],
                      "The 1st dimension of Input(Predictions) and "
                      "Input(Labels) both are batch_size and the shape should "
                      "be the same.");
    PADDLE_ENFORCE_EQ(labels_dims[1], 1,
                      "The 2nd dimension of Input(Labels) "
                      "contains instance label and the shape should be equal "
                      "to 1");
    PADDLE_ENFORCE_GE(predictions_dims[1], 1,
                      "The shape of Input(Predictions)'s 2nd dimension is "
                      "equal to class number and should be at least 1.");

    // Layouts of BatchMetrics and AccumMetrics both are:
    // [
    //  macro average precision, macro average recall, macro average F1 score,
    //  micro average precision, micro average recall, micro average F1 score
    // ]
    ctx->SetOutputDim("BatchMetrics", {6});
    ctx->SetOutputDim("AccumMetrics", {6});
    // Shape of AccumStatesInfo is [class_number, 4]
    // The layout of each row is:
    // [ TP, FP, TN, FN ]
    ctx->SetOutputDim("AccumStatesInfo", {predictions_dims[1], 4});
  }

 protected:
  framework::DataType IndicateDataType(
      const framework::ExecutionContext &ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("Predictions")->type());
  }
};

class PrecisionRecallOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PrecisionRecallOpMaker(framework::OpProto *proto,
                         framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Predictions",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape N x D, "
             "where N is the batch size and D is the number of classes. "
             "Each row contains probabilities for an instance which computed "
             "by the previous operator.");
    AddInput("Labels",
             "(Tensor, default Tensor<int>), a 2-D tensor with shape N x 1, "
             "where N is the batch size. Each element is a label and the "
             "value should be in [0, class_number - 1].");
    AddInput("Weights",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape N x 1, "
             "where N is the batch size. This input is optional. If provided, "
             "weight of instance would be considered when computing metrics.")
        .AsDispensable();
    AddInput("StatesInfo",
             "(Tensor, default Tensor<int>), a 2-D tensor with shape D x 4, "
             "where D is the number of classes. This input is optional. If "
             "provided, current state will be accumulated to this state and "
             "the accumulation state will be as the output state.")
        .AsDispensable();
    AddOutput("BatchMetrics",
              "(Tensor, default Tensor<float>), a 1-D tensor with shape {6}."
              "This output tensor contains metrics for current batch data."
              "The layout is [macro average precision, macro average recall, "
              "macro f1 score, micro average precision, micro average recall, "
              "micro f1 score]");
    AddOutput("AccumMetrics",
              "(Tensor, default Tensor<float>), a 1-D tensor with shape {6}."
              "This output tensor contains metrics for accumulated data."
              "The layout is [macro average precision, macro average recall, "
              "macro f1 score, micro average precision, micro average recall, "
              "micro f1 score]");
    AddOutput("AccumStatesInfo",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape D x 4, "
              "where D is equal to class number. This output tensor contains "
              "accumulated state variables used to compute metrics. The layout "
              "for each class is [true positives, false positives, "
              "true negatives, false negatives].");

    AddComment(R"DOC(
When given 'Input(Predictions)' and 'Input(Labels)', this operator can be used
to compute various metrics including:
  - macro average precision
  - macro average recall
  - macro f1 score
  - micro average precision
  - micro average recall
  - micro f1 score

To compute the above metrics, we need to do statistics for true positives,
false positives and false negatives. Here count of true negatives is not
necessary, but counting it may provide potential usage and the cost is
trivial, so the operator also provides count of true negatives.

We define state as a 2-D tensor with shape [class number, 4]. Each row of a
state contains statistic variables for corresponding class. Layout of each row
is: TP(true positives), FP(false positives), TN(true negatives),
FN(false negatives). If 'Input(Weights)' provided, TP, FP, TN, FN will be
calculated by given weight instead of instance count.

This operator also supports metrics computing for cross-batch situation. To
achieve this, 'Input(StatesInfo)' should be provided. State of current batch
data will be accumulated to 'Input(StatesInfo)' and 'Output(AccumStatesInfo)'
is the accumulation state.

'Output(BatchMetrics)' is metrics of current batch data while
'Output(AccumStatesInfo)' is metrics of accumulation data.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(precision_recall, ops::PrecisionRecallOp,
                             ops::PrecisionRecallOpMaker);
REGISTER_OP_CPU_KERNEL(
    precision_recall,
    ops::PrecisionRecallKernel<paddle::platform::CPUPlace, float>,
    ops::PrecisionRecallKernel<paddle::platform::CPUPlace, double>);
