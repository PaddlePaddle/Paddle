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

#include "paddle/fluid/operators/positive_negative_pair_op.h"

namespace paddle {
namespace operators {

class PositiveNegativePairOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Score"),
        "Input(Score) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Label"),
        "Input(Label) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("QueryID"),
        "Input(QueryID) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("PositivePair"),
        "Output(PositivePair) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("NegativePair"),
        "Output(NegativePair) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("NeutralPair"),
        "Output(NeutralPair) of PositiveNegativePairOp should not be null.");
    auto scalar_dim = framework::make_ddim({1});
    if (ctx->HasInput("AccumulatePositivePair") ||
        ctx->HasInput("AccumulateNegativePair") ||
        ctx->HasInput("AccumulateNeutralPair")) {
      PADDLE_ENFORCE(ctx->HasInput("AccumulatePositivePair") &&
                         ctx->HasInput("AccumulateNegativePair") &&
                         ctx->HasInput("AccumulateNeutralPair"),
                     "All optional inputs(AccumulatePositivePair, "
                     "AccumulateNegativePair, AccumulateNeutralPair) of "
                     "PositiveNegativePairOp are required if one of them is "
                     "specified.");
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("AccumulatePositivePair"), scalar_dim,
                        "Shape of AccumulatePositivePair should be {1}.");
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("AccumulateNegativePair"), scalar_dim,
                        "Shape of AccumulateNegativePair should be {1}.");
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("AccumulateNeutralPair"), scalar_dim,
                        "Shape of AccumulateNeutralPair should be {1}.");
    }

    auto score_dim = ctx->GetInputDim("Score");
    auto label_dim = ctx->GetInputDim("Label");
    auto query_dim = ctx->GetInputDim("QueryID");
    PADDLE_ENFORCE_EQ(score_dim.size(), 2, "Score should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(label_dim.size(), 2, "Label should be a 2-D tensor.");

    if (ctx->IsRuntime() ||
        (score_dim[0] > 0 && label_dim[0] > 0 && query_dim[0] > 0)) {
      PADDLE_ENFORCE_EQ(
          label_dim[0], score_dim[0],
          "Tensor Score and Label should have the same height (batch size).");

      PADDLE_ENFORCE_EQ(label_dim[1], 1,
                        "The width of Label should be 1, i.e. each item should "
                        "have a scalar label.");

      PADDLE_ENFORCE(query_dim == label_dim,
                     "QueryID should have the same shape as Label.");

      if (ctx->HasInput("Weight")) {
        PADDLE_ENFORCE(ctx->GetInputDim("Weight") == label_dim,
                       "Weight should have the same shape as Label.");
      }

      int column = ctx->Attrs().Get<int>("column");
      auto depth = score_dim[1];
      PADDLE_ENFORCE(column < depth && column >= -depth,
                     "Attribute column should be in the range of [-%l, %l)",
                     depth, depth);
    }

    ctx->SetOutputDim("PositivePair", scalar_dim);
    ctx->SetOutputDim("NegativePair", scalar_dim);
    ctx->SetOutputDim("NeutralPair", scalar_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Score"),
        ctx.device_context());
  }
};

class PositiveNegativePairOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Score",
             "(Tensor, float) Model Score on an item (with "
             "respect to QueryID). It's a 2-D tensor with shape [batch_size, "
             "depth], where the column specified by the attribute \"column\" "
             "is used as item score.");
    AddInput("Label",
             "(Tensor, float) Label of an item (with repsect to "
             "QueryId). It's a 2-D tensor with shape [batch_size, 1].");
    AddInput("QueryID",
             "(Tensor, int64) Query ID that indicates the context. Its shape "
             "should be the same as Label.");
    AddInput(
        "AccumulatePositivePair",
        "(float) Optional. The accumulated number of positive pairs over a "
        "stream of data. If provided, the output PositivePair will be "
        "initialized with this number rather than 0. it won't be modified "
        "in place.")
        .AsDispensable();
    AddInput(
        "AccumulateNegativePair",
        "(float) Optional. The accumulated number of negative pairs over a "
        "stream of data. If provided, the output NegativePair will be "
        "initialized with this number rather than 0. it won't be modified "
        "in place.")
        .AsDispensable();
    AddInput("AccumulateNeutralPair",
             "(float) Optional. The accumulated number of neutral pairs over a "
             "stream of data. If provided, the output NeutralPair will be "
             "initialized with this number rather than 0. it won't be modified "
             "in place.")
        .AsDispensable();
    AddInput("Weight",
             "(float) Optional. Weight of current item. If specified, its "
             "shape should be the same as Label, and the meaning of the output "
             "changes from numbers of pairs to the total sum of pairs' "
             "weights. Weight of a pair of items is the average of their "
             "weights.")
        .AsDispensable();
    AddOutput("PositivePair",
              "(float) Number of positive pairs, i.e. the pairs of "
              "items that are ranked correctly.");
    AddOutput("NegativePair",
              "(float) Number of negative pairs, i.e. the pairs of "
              "items that are ranked incorrectly.");
    AddOutput("NeutralPair",
              "(float) Number of neutral pairs, i.e. the pairs of items "
              "that have the same score.")
        .AsDispensable();
    AddAttr<int>(
        "column",
        "(int, default -1) The column position of Score used to rank items in "
        "descending order. It must be in the range of [-rank(Score), "
        "rank(Score)). "
        "If `dim < 0`, the dim to reduce is `rank + dim`. "
        "Noting that reducing on the first dim will make the LoD info lost.")
        .SetDefault(0);
    AddComment(R"DOC(
PositiveNegativePairOp can be used to evaluate Learning To Rank(LTR) model's
performance.

Within some context, e.g. the "query", a LTR model generates scores for a list
of items, which gives a partial order of the items. PositiveNegativePairOp
takes a list of reference rank order (Input("Label")) and the model generated
scores (Input(Score)) as inputs and counts the pairs that ranked correctly
and incorrectly.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(positive_negative_pair,
                             ops::PositiveNegativePairOp,
                             ops::PositiveNegativePairOpMaker);
REGISTER_OP_CPU_KERNEL(
    positive_negative_pair,
    ops::PositiveNegativePairKernel<paddle::platform::CPUPlace, float>,
    ops::PositiveNegativePairKernel<paddle::platform::CPUPlace, double>);
