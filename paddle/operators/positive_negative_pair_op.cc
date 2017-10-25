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

#include "paddle/operators/positive_negative_pair_op.h"

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
        ctx->HasInput("QueryId"),
        "Input(QueryId) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("PositivePair"),
        "Output(PositivePair) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("NegativePair"),
        "Output(NegativePair) of PositiveNegativePairOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("NeutralPair"),
        "Output(NeutralPair) of PositiveNegativePairOp should not be null.");

    auto score_dim = ctx->GetInputDim("Score");
    auto label_dim = ctx->GetInputDim("Label");
    auto query_dim = ctx->GetInputDim("QueryId");

    PADDLE_ENFORCE(score_dim == label_dim,
                   "Shape of Score must be the same as Label's shape.");
    PADDLE_ENFORCE(query_dim == label_dim,
                   "Shape of QueryId must be the same as Label's shape.");
    PADDLE_ENFORCE(query_dim == label_dim,
                   "Shape of QueryId must be the same as Label's shape.");

    ctx->SetOutputDim("PositivePair", {1});
    ctx->SetOutputDim("NegativePair", {1});
    ctx->SetOutputDim("NeutralPair", {1});
  }

 protected:
  framework::DataType IndicateDataType(
      const framework::ExecutionContext &ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("Score")->type());
  }
};

class PositiveNegativePairOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PositiveNegativePairOpMaker(framework::OpProto *proto,
                              framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Score",
             "(Tensor, float) Output score of the network on <query, document> "
             "pair.");
    AddInput("Label",
             "(Tensor, float or int) Label of current <query, document> pair.");
    AddInput("QueryId",
             "(Tensor, int) query id of current <query, document> pair.");
    AddOutput("PositivePair",
              "(float) Number of positive ranking pairs, i.e. the pairs of "
              "documents that are ranked correctly");
    AddOutput("NegativePair",
              "(float) Number of negative ranking pairs, i.e. the pairs of "
              "documents that are ranked incorrectly");
    AddOutput("NeutralPair",
              "(float) Number of neutral ranking pairs. A pair of document "
              "(doc#1, doc#2) is classified as \"neutral\" if their scores are "
              "the same.");
    AddComment(R"DOC(
        PositiveNegativePairOp can be used to evaluate Learning To Rank(LTR) model performance. Its outputs are usually 
        further summarized as positive-negative-ratio: PositivePair/NegativePair.
        Its 3 inputs can be viewd as a series of 3 tuples: (predicition score, golden label, query id).
        For each unique query id, a list of <score, label> are collected and positive/negative pairs are accumulated to its output. 
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
    ops::PositiveNegativePairKernel<paddle::platform::CPUPlace, float>);
