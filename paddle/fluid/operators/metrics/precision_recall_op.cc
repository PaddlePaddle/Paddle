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

#include "paddle/fluid/operators/metrics/precision_recall_op.h"

namespace paddle {
namespace operators {

class PrecisionRecallOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("MaxProbs"), true,
        platform::errors::NotFound(
            "PrecisionRecallOp Input(MaxProbs) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Indices"), true,
        platform::errors::NotFound(
            "PrecisionRecallOp Input(Indices) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Labels"), true,
        platform::errors::NotFound(
            "PrecisionRecallOp Input(Labels) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchMetrics"), true,
        platform::errors::NotFound(
            "PrecisionRecallOp Output(BatchMetrics) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("AccumMetrics"), true,
        platform::errors::NotFound(
            "PrecisionRecallOp Output(AccumMetrics) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("AccumStatesInfo"), true,
        platform::errors::NotFound(
            "PrecisionRecallOp Output(AccumStatesInfo) should not be null."));

    int64_t cls_num =
        static_cast<int64_t>(ctx->Attrs().Get<int>("class_number"));
    auto max_probs_dims = ctx->GetInputDim("MaxProbs");
    auto labels_dims = ctx->GetInputDim("Labels");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(max_probs_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "Each instance of PrecisionRecallOp "
                            "Input(MaxProbs) contains one max probability, "
                            "the shape of Input(MaxProbs) should be "
                            "[batch_size, 1], the 2nd dimension of "
                            "Input(MaxProbs) should be 1. But the 2nd "
                            "dimension we received is %d",
                            max_probs_dims[1]));
      PADDLE_ENFORCE_EQ(
          ctx->GetInputDim("Indices"), max_probs_dims,
          platform::errors::InvalidArgument(
              "The shape of PrecisionRecallOp Input(Indices) should be same "
              "with "
              "max_probs_dims. But received the shape of Input(Indices) is "
              "[%d, %d], max_probs_dims is [%d, %d]",
              ctx->GetInputDim("Indices")[0], ctx->GetInputDim("Indices")[1],
              max_probs_dims[0], max_probs_dims[1]));
      PADDLE_ENFORCE_EQ(
          max_probs_dims[0], labels_dims[0],
          platform::errors::InvalidArgument(
              "The 1st dimension of PrecisionRecallOp Input(MaxProbs) and "
              "Input(Labels) both should be batch_size"
              "But the 1st dimension we received max_probs_dims[0] = %d, "
              "labels_dims[0] = %d",
              max_probs_dims[0], labels_dims[0]));
      PADDLE_ENFORCE_EQ(labels_dims[1], 1,
                        platform::errors::InvalidArgument(
                            "The 2nd dimension of PrecisionRecallOp "
                            "Input(Labels) contains instance label and "
                            "the shape should be equal to 1. But the 2nd "
                            "dimension we received is %d",
                            labels_dims[1]));
    }
    if (ctx->HasInput("Weights")) {
      auto weights_dims = ctx->GetInputDim("Weights");

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            weights_dims, phi::make_ddim({max_probs_dims[0], 1}),
            platform::errors::InvalidArgument(
                "The shape of PrecisionRecallOp Input(Weights) should be "
                "[batch_size, 1]. But the shape we received is [%d, %d]",
                weights_dims[0], weights_dims[1]));
      }
    }
    if (ctx->HasInput("StatesInfo")) {
      auto states_dims = ctx->GetInputDim("StatesInfo");

      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            states_dims, phi::make_ddim({cls_num, 4}),
            platform::errors::InvalidArgument(
                "The shape of PrecisionRecallOp Input(StatesInfo) should be "
                "[class_number, 4]. But the shape we received is [%d, %d]",
                states_dims[0], states_dims[1]));
      }
    }

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
    ctx->SetOutputDim("AccumStatesInfo", {cls_num, 4});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "MaxProbs"),
        ctx.device_context());
  }
};

class PrecisionRecallOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("MaxProbs",
             "(Tensor, default Tensor<float>) A 2-D tensor with shape N x 1, "
             "where N is the batch size. Each row contains the max probability "
             "of an instance which computed by the previous top_k (k=1) "
             "operator.");
    AddInput("Indices",
             "(Tensor, default Tensor<int>) A 2-D tensor with shape N x 1, "
             "where N is the batch size. Each row contains the corresponding "
             "index which computed by the previous top_k (k=1) operator.");
    AddInput("Labels",
             "(Tensor, default Tensor<int>) A 2-D tensor with shape N x 1, "
             "where N is the batch size. Each element is a label and the "
             "value should be in [0, class_number - 1].");
    AddInput("Weights",
             "(Tensor, default Tensor<float>) A 2-D tensor with shape N x 1, "
             "where N is the batch size. This input is optional. If provided, "
             "weight of instance would be considered when computing metrics.")
        .AsDispensable();
    AddInput("StatesInfo",
             "(Tensor, default Tensor<int>) A 2-D tensor with shape D x 4, "
             "where D is the number of classes. This input is optional. If "
             "provided, current state will be accumulated to this state and "
             "the accumulation state will be the output state.")
        .AsDispensable();
    AddOutput("BatchMetrics",
              "(Tensor, default Tensor<float>) A 1-D tensor with shape {6}. "
              "This output tensor contains metrics for current batch data. "
              "The layout is [macro average precision, macro average recall, "
              "macro f1 score, micro average precision, micro average recall, "
              "micro f1 score].");
    AddOutput("AccumMetrics",
              "(Tensor, default Tensor<float>) A 1-D tensor with shape {6}. "
              "This output tensor contains metrics for accumulated data. "
              "The layout is [macro average precision, macro average recall, "
              "macro f1 score, micro average precision, micro average recall, "
              "micro f1 score].");
    AddOutput("AccumStatesInfo",
              "(Tensor, default Tensor<float>) A 2-D tensor with shape D x 4, "
              "where D is equal to class number. This output tensor contains "
              "accumulated state variables used to compute metrics. The layout "
              "for each class is [true positives, false positives, "
              "true negatives, false negatives].");
    AddAttr<int>("class_number", "(int) Number of classes to be evaluated.");
    AddComment(R"DOC(
Precision Recall Operator.

When given Input(Indices) and Input(Labels), this operator can be used
to compute various metrics including:
1. macro average precision
2. macro average recall
3. macro f1 score
4. micro average precision
5. micro average recall
6. micro f1 score

To compute the above metrics, we need to do statistics for true positives,
false positives and false negatives. Here the count of true negatives is not
necessary, but counting it may provide potential usage and the cost is
trivial, so the operator also provides the count of true negatives.

We define state as a 2-D tensor with shape [class_number, 4]. Each row of a
state contains statistic variables for corresponding class. Layout of each row
is: TP(true positives), FP(false positives), TN(true negatives),
FN(false negatives). If Input(Weights) is provided, TP, FP, TN, FN will be
calculated by given weight instead of the instance count.

This operator also supports metrics computing for cross-batch situation. To
achieve this, Input(StatesInfo) should be provided. State of current batch
data will be accumulated to Input(StatesInfo) and Output(AccumStatesInfo)
is the accumulation state.

Output(BatchMetrics) is metrics of current batch data while
Output(AccumStatesInfo) is metrics of accumulation data.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    precision_recall, ops::PrecisionRecallOp, ops::PrecisionRecallOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    precision_recall,
    ops::PrecisionRecallKernel<paddle::platform::CPUPlace, float>,
    ops::PrecisionRecallKernel<paddle::platform::CPUPlace, double>);
