/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class FusedTokenPruneOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Attn",
             "(Tensor)"
             "The input of fused_token_prune op, whole shape should be [bsz, 12, max_seq_len, max_seq_len] and dtype should be float32/float64"
             "Attn is attention scores of input sequences which will be used to sort another input tensor: X's ");

    AddInput("X",
             "(Tensor)"
             "[bsz, max_seq_len, 768] float 32 layer_norm_3.tmp_2");

    AddInput("Mask",
             "(Tensor)"
             "[bsz, 12, max_seq_len, max_seq_len] float32 stack_0.tmp_0");

    AddInput(
        "NewMask",
        "(Tensor)"
        "[bsz, 12, slimmed_seq_len, slimmed_seq_len] float32 stack_0.tmp_0");

    AddOutput("SlimmedX",
              "(Tensor)"
              "[bsz, max_seq_len * factor, C = 768]");

    AddComment(R"DOC(
                            Operator.
                        )DOC");
  }
};

class FusedTokenPruneOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Attn"), "Input", "Attn", "FusedTokenPrune");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedTokenPrune");
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "FusedTokenPrune");
    OP_INOUT_CHECK(ctx->HasInput("NewMask"), "Input", "NewMask",
                   "FusedTokenPrune");
    OP_INOUT_CHECK(ctx->HasOutput("SlimmedX"), "Output", "SlimmedX",
                   "FusedTokenPrune");

    auto mask_dim = ctx->GetInputDim("Mask");
    auto attn_dim = ctx->GetInputDim("Attn");
    auto x_dim = ctx->GetInputDim("X");
    auto new_mask_dim = ctx->GetInputDim("NewMask");

    // check input dims number
    PADDLE_ENFORCE_EQ(mask_dim.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input mask must be 4-dimention"));
    PADDLE_ENFORCE_EQ(attn_dim.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input attn must be 4-dimention"));
    PADDLE_ENFORCE_EQ(x_dim.size(), 3, platform::errors::InvalidArgument(
                                           "The input x must be 4-dimention"));
    PADDLE_ENFORCE_EQ(new_mask_dim.size(), 4,
                      platform::errors::InvalidArgument(
                          "The input attn must be 4-dimention"));

    // check input dims relations
    PADDLE_ENFORCE_EQ(mask_dim[0], attn_dim[0],
                      platform::errors::InvalidArgument(
                          "The first dim of mask and attn should be the same"
                          "which is batch size"));
    PADDLE_ENFORCE_EQ(mask_dim[1], attn_dim[1],
                      platform::errors::InvalidArgument(
                          "The second dim of mask and attn should be the same"
                          "which is nb_head"));
    PADDLE_ENFORCE_EQ(mask_dim[0], x_dim[0],
                      platform::errors::InvalidArgument(
                          "The first dim of mask and x should be the same"
                          "which is batch size"));
    PADDLE_ENFORCE_EQ(
        mask_dim[2], mask_dim[3],
        platform::errors::InvalidArgument(
            "The third dim and the fourth dim of mask should be the same"
            "which is max seq len"));
    PADDLE_ENFORCE_EQ(
        attn_dim[2], attn_dim[3],
        platform::errors::InvalidArgument(
            "The third dim and the fourth dim of mask should be the same"
            "which is max seq len"));
    PADDLE_ENFORCE_EQ(attn_dim[2], mask_dim[2],
                      platform::errors::InvalidArgument(
                          "The third dim of mask and attn should be the same"
                          "which is max seq len"));
    PADDLE_ENFORCE_EQ(attn_dim[2], x_dim[1],
                      platform::errors::InvalidArgument(
                          "The third dim of mask and the second dim of attn"
                          "should be the same which is max seq len"));

    auto bsz = mask_dim[0];
    auto c = x_dim[2];
    auto slim_seq_len = new_mask_dim[2];

    ctx->SetOutputDim("SlimmedX", {bsz, slim_seq_len, c});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fused_token_prune, ops::FusedTokenPruneOp, ops::FusedTokenPruneOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
