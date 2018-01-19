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

#include "paddle/operators/mine_hard_examples_op.h"

namespace paddle {
namespace operators {

class MineHardExamplesOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("ClsLoss"),
                   "Input(ClsLoss) of MineHardExamplesOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("MatchIndics"),
        "Input(MatchIndics) of MineHardExamplesOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("MatchDis"),
                   "Input(MatchDis) of MineHardExamplesOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("NegIndics"),
        "Output(NegIndics) of MineHardExamplesOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("UpdatedMatchIndics"),
        "Output(UpdatedMatchIndics) of MineHardExamplesOp should not be null.");

    auto cls_loss_dims = ctx->GetInputDim("ClsLoss");
    auto idx_dims = ctx->GetInputDim("MatchIndics");
    auto dis_dims = ctx->GetInputDim("MatchDis");

    PADDLE_ENFORCE_EQ(cls_loss_dims.size(), 2UL,
                      "The shape of ClsLoss is [N, Np].");
    PADDLE_ENFORCE_EQ(idx_dims.size(), 2UL,
                      "The shape of MatchIndics is [N, Np].");
    PADDLE_ENFORCE_EQ(dis_dims.size(), 2UL,
                      "The shape of MatchDis is [N, Np].");

    if (ctx->HasInput("LocLoss")) {
      auto loc_loss_dims = ctx->GetInputDim("LocLoss");
      PADDLE_ENFORCE_EQ(loc_loss_dims.size(), 2UL,
                        "The shape of LocLoss is [N, Np].");
      PADDLE_ENFORCE_EQ(cls_loss_dims[0], loc_loss_dims[0],
                        "Batch size of ClsLoss and LocLoss must be the same.");
      PADDLE_ENFORCE_EQ(
          cls_loss_dims[1], loc_loss_dims[1],
          "Prior box number of ClsLoss and LocLoss must be the same.");
    }

    PADDLE_ENFORCE_EQ(
        cls_loss_dims[0], idx_dims[0],
        "Batch size of ClsLoss and MatchIndics must be the same.");
    PADDLE_ENFORCE_EQ(
        cls_loss_dims[1], idx_dims[1],
        "Prior box number of ClsLoss and MatchIndics must be the same.");

    PADDLE_ENFORCE_EQ(cls_loss_dims[0], dis_dims[0],
                      "Batch size of ClsLoss and MatchDis must be the same.");
    PADDLE_ENFORCE_EQ(
        cls_loss_dims[1], idx_dims[1],
        "Prior box number of ClsLoss and MatchDis must be the same.");

    auto mining_type =
        GetMiningType(ctx->Attrs().Get<std::string>("mining_type"));

    PADDLE_ENFORCE_NE(mining_type, MiningType::kNone,
                      "mining_type must be hard_example or max_negative");

    if (mining_type == MiningType::kMaxNegative) {
      auto neg_pos_ratio = ctx->Attrs().Get<float>("neg_pos_ratio");
      auto neg_dis_threshold = ctx->Attrs().Get<float>("neg_dis_threshold");
      PADDLE_ENFORCE_GT(
          neg_pos_ratio, 0.0f,
          "neg_pos_ratio must greater than zero in max_negative mode");
      PADDLE_ENFORCE_GT(
          neg_dis_threshold, 0.0f,
          "neg_dis_threshold must greater than zero in max_negative mode");
    } else if (mining_type == MiningType::kHardExample) {
      auto sample_size = ctx->Attrs().Get<int>("sample_size");
      PADDLE_ENFORCE_GT(
          sample_size, 0,
          "sample_size must greater than zero in hard_example mode");
    }

    ctx->SetOutputDim("UpdatedMatchIndics", idx_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("ClsLoss")->type()),
        ctx.device_context());
  }
};

class MineHardExamplesOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MineHardExamplesOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "ClsLoss",
        "(Tensor, default Tensor<float>), The classification loss wit shape "
        "[N, Np], N is the batch size and Np is the number of prior box.");
    AddInput("LocLoss",
             "(Tensor, optional, default Tensor<float>), The localization loss "
             "wit shape [N, Np], N is the batch size and Np is the number of "
             "prior box.")
        .AsDispensable();
    AddInput("MatchIndics",
             "(Tensor, Tensor<int>), Matched indices with shape [N, Np], N is "
             "the batch size and Np is the number of prior box. "
             "MatchIndics[i][j] equal -1 means box[j] does not match any "
             "entity, otherwise means Box[j] is matched to row.");
    AddInput("MatchDis",
             "(Tensor, default Tensor<float>) Matched indices with shape [N, "
             "Np], N is the batch size and Np is the number of prior box.");
    AddAttr<float>("neg_pos_ratio",
                   "(float) The ratio of the negative box to the positive "
                   "box. Use only when mining_type is equal to max_negative.")
        .SetDefault(1.0);
    AddAttr<float>("neg_dis_threshold",
                   "(float) The negative box dis value threshold. "
                   "Use only when mining_type is equal to max_negative.")
        .SetDefault(0.5);
    AddAttr<int>("sample_size",
                 "(float) The max sample size of negative box. Use only when "
                 "mining_type is equal to hard_example.")
        .SetDefault(0);
    AddAttr<std::string>("mining_type",
                         "(float) The mining algorithm name, the value is "
                         "hard_example or max_negative.")
        .SetDefault("max_negative")
        .InEnum({"hard_example", "max_negative"});

    AddOutput("NegIndics",
              "(LoDTensor) The output of negative example indics.a lod tensor "
              "with shape [Neg, 1]. The size of lod[0] is batch size, "
              "and each element is the box index. "
              "For example, the batch size is 2, the lod is [[0, 1, 2]], "
              "the sample 0's box 1(MatchIndics[0][1]) is selected, "
              "and sample 1's box 0 is selected. The output NegIndics is "
              "[[1], [0]].");

    AddOutput("UpdatedMatchIndics",
              "(Tensor) The output of updated MatchIndics, a tensor with "
              "shape [N, M]. Only update when mining_type is equal to "
              "hard_example. The input MatchIndics elements will be update to "
              "-1 when it not in the highest loss list");

    AddComment(R"DOC(
Mine hard examples Operator.
This operator implements hard example mining to select a subset of negative box indics.
For each image, selects the box with highest losses. subject to the condition that the box cannot have
an MatchDis > neg_dis_threshold when mining_type is equals max_negative. The selected number is 
min(sample_size, max_negative_box_number) when mining_type is equals hard_example,
or min(neg_pos_ratio * positive_box_number, max_negative_box_number) when mining_type is 
equals max_negative, where the max_negative_box_number is the count of MatchIndics elements with value -1.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(mine_hard_examples, ops::MineHardExamplesOp,
                             ops::MineHardExamplesOpMaker);

REGISTER_OP_CPU_KERNEL(
    mine_hard_examples,
    ops::MineHardExamplesKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MineHardExamplesKernel<paddle::platform::CPUDeviceContext, double>);
