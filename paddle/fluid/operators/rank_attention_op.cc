/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/rank_attention_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

class RankAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(X) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("RankOffset"),
        true,
        platform::errors::InvalidArgument(
            "Input(RankOffset) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("RankParam"),
        true,
        platform::errors::InvalidArgument(
            "Input(RankParam) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("InsRank"),
        true,
        platform::errors::InvalidArgument(
            "Output(InsRank) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("InputHelp"),
        true,
        platform::errors::InvalidArgument(
            "Output(InputHelp) of RankAttentionOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"),
        true,
        platform::errors::InvalidArgument(
            "Output(Out) of RankAttentionOp should not be null."));
    auto max_rank = ctx->Attrs().Get<int>("MaxRank");

    auto x_dims = ctx->GetInputDim("X");
    auto ins_num = x_dims[0];
    auto param_dims = ctx->GetInputDim("RankParam");
    auto para_col = param_dims[1];
    auto rank_offset_dims = ctx->GetInputDim("RankOffset");
    auto x_fea_dim = x_dims[1];
    auto block_matrix_row = max_rank * x_fea_dim;

    PADDLE_ENFORCE_EQ((rank_offset_dims[1] - 1) / 2,
                      max_rank,
                      platform::errors::InvalidArgument(
                          "Input(RankOffset) has wrong columns, "
                          "except columns to be %d, but got %d",
                          max_rank,
                          (rank_offset_dims[1] - 1) / 2));

    ctx->SetOutputDim("Out", {ins_num, para_col});
    ctx->SetOutputDim("InputHelp", {ins_num, block_matrix_row});
    ctx->SetOutputDim("InsRank", {ins_num, 1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class RankAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"),
        true,
        platform::errors::InvalidArgument("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("RankParam"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(RankParam) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("RankOffset"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(RankOffset) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("InputHelp"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(InputHelp) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("InsRank"),
        true,
        platform::errors::InvalidArgument("Input(InsRank) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("RankParam"),
                      ctx->GetInputDim("RankParam"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class RankAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of rank_attention_Op operator.");
    AddInput("RankOffset",
             "(Tensor) Input tensor of rank_attention_Op operator.");
    AddInput("RankParam",
             "(Tensor) Input tensor of rank_attention_Op operator.");
    AddOutput("InputHelp", "Output tensor of rank_attention_Op operator.")
        .AsDispensable();
    AddOutput("Out", "Output tensor of rank_attention_Op operator.");
    AddOutput("InsRank", "Output tensor of rank_attention_Op operator.")
        .AsDispensable();
    AddAttr<int>("MaxRank", "(int, default 3) max rank of rank_attention_Op")
        .SetDefault(3);
    AddAttr<int>("MaxSize", "(int, default 0) max rank of rank_attention_Op")
        .SetDefault(0);
    AddComment(R"DOC(
RankAttention Operator.
This Op can calculate rank attention between input and rank_param,
and rank_param gives the organization of data. Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

template <typename T>
class RankAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("rank_attention_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("RankOffset", this->Input("RankOffset"));
    op->SetInput("RankParam", this->Input("RankParam"));
    op->SetInput("InputHelp", this->Output("InputHelp"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("InsRank", this->Output("InsRank"));

    op->SetOutput(framework::GradVarName("RankParam"),
                  this->InputGrad("RankParam"));
    op->SetAttrMap(this->Attrs());
  }
};
DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    RankAttentionGradOpNoNeedBufferVarsInference,
    "X",
    "RankOffset",
    "RankParam");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(rank_attention,
                  ops::RankAttentionOp,
                  ops::RankAttentionOpMaker,
                  ops::RankAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::RankAttentionGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(rank_attention_grad,
                  ops::RankAttentionGradOp,
                  ops::RankAttentionGradOpNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(rank_attention,
                       ops::RankAttentionKernel<phi::CPUContext, float>,
                       ops::RankAttentionKernel<phi::CPUContext, double>);

REGISTER_OP_VERSION(rank_attention)
    .AddCheckpoint(
        R"ROC(
        Upgrade rank_attention, add 1 outputs [InputHelp] and 1 attribute
        [MaxSize].
      )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewOutput("InputHelp",
                       "Output tensor of rank_attention_Op operator "
                       "in order to assist calculation in the reverse process.")
            .NewAttr(
                "MaxSize",
                "Forward calculation to set the pre-applied video memory size",
                0));
