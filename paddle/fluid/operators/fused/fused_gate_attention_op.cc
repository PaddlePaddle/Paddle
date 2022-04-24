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

#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FusedGateAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasInput("QKVWeight"), "Input", "QKVWeight",
                   "FusedGateAttention");

    OP_INOUT_CHECK(ctx->HasInput("OutLinearWeight"), "Input", "OutLinearWeight",
                   "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearBias"), "Input", "OutLinearBias",
                   "FusedGateAttention");

    OP_INOUT_CHECK(ctx->HasOutput("QKVOut"), "Output", "QKVOut",
                   "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasOutput("QKVTransposeOut"), "Output",
                   "QKVTransposeOut", "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasOutput("QKOut"), "Output", "QKOut",
                   "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasOutput("QKTVOut"), "Output", "QKTVOut",
                   "FusedGateAttention");

    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxOut"), "Output", "SoftmaxOut",
                   "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasOutput("FMHAOut"), "Output", "FMHAOut",
                   "FusedGateAttention");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "FusedGateAttention");

    // x: qkv's input [batch_size, seq_len_m, seq_len_r, c]
    // y: qkv's weight: [3, n_head, c, qkv_dim]
    auto input_x_dims = ctx->GetInputDim("X");
    auto qkv_w_dims = ctx->GetInputDim("QKVWeight");

    int batch_size = input_x_dims[0];
    int seq_len_m = input_x_dims[1];
    int seq_len_r = input_x_dims[2];

    // qkv_weight[3, n_head, c, qkv_dim]
    int num_head = qkv_w_dims[1];
    int c = qkv_w_dims[2];

    ctx->SetOutputDim("QKVOut",
                      {batch_size, seq_len_m, seq_len_r, 3, num_head, c});
    ctx->SetOutputDim("QKVTransposeOut",
                      {3, batch_size, seq_len_m, num_head, seq_len_r, c});

    ctx->SetOutputDim("QKOut",
                      {batch_size, seq_len_m, num_head, seq_len_r, seq_len_r});
    ctx->SetOutputDim("SoftmaxOut",
                      {batch_size, seq_len_m, num_head, seq_len_r, seq_len_r});

    ctx->SetOutputDim("QKTVOut",
                      {batch_size, seq_len_m, num_head, seq_len_r, c});
    ctx->SetOutputDim("FMHAOut",
                      {batch_size, seq_len_m, seq_len_r, num_head, c});

    if (ctx->Attrs().Get<bool>("is_gating")) {
      ctx->SetOutputDim("GateBiasOut",
                        {batch_size, seq_len_m, seq_len_r, num_head, c});
      ctx->SetOutputDim("GateOut",
                        {batch_size, seq_len_m, seq_len_r, num_head, c});
    }

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class FusedGateAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("QKVWeight", "The qkv weight tensor.");
    AddInput("NonbatchedBias", "(optional) The nonbatchedBias .")
        .AsDispensable();
    AddInput("SrcMask", "The attention mask tensor in fmha.");
    AddInput("GateWeight", "(optional) The gate weight tensor.")
        .AsDispensable();
    AddInput("GateBias", "(optional) The gate bias tensor.").AsDispensable();
    AddInput("OutLinearWeight", "The out_linear weight tensor.");
    AddInput("OutLinearBias", "The out_linear bias tensor.");
    AddOutput("QKVOut", "Result after qkv.").AsIntermediate();
    AddOutput("QKVTransposeOut", "Result in fmha.").AsIntermediate();
    AddOutput("QKOut", "Result in fmha.").AsIntermediate();
    AddOutput("QKTVOut", "Result in fmha.").AsIntermediate();
    AddOutput("SoftmaxOut", "Result in fmha.").AsIntermediate();
    AddOutput("FMHAOut", "Result after fmha.").AsIntermediate();
    AddOutput("GateBiasOut", "Result after add bias")
        .AsIntermediate()
        .AsDispensable();
    AddOutput("GateOut", "Result gate").AsIntermediate().AsDispensable();
    AddOutput("Y", "Result after attention.");
    AddAttr<bool>("is_gating",
                  "if true, the attention op uses gate architecure, "
                  "[default true].")
        .SetDefault(true);

    AddComment(R"DOC(
  Add fused attention op whose logic is as follows:
  {
    q = paddle.einsum('nbqa,ahc->nbqhc', q_data, self.query_w) 
    k = paddle.einsum('nbka,ahc->nbkhc', m_data, self.key_w)
    v = paddle.einsum('nbka,ahc->nbkhc', m_data, self.value_w)
    logits = paddle.einsum('nbqhc,nbkhc->nbhqk', q * c , k) + bias
    weights = nn.functional.softmax(logits)
    weighted_avg = paddle.einsum('nbhqk,nbkhc->nbqhc', weights, v)
    if nonbatched_bias is not None:
      nonbatched_bias = all_gather_opp(nonbatched_bias, axis=2, sync=self.comm_sync)
      logits += paddle.unsqueeze(nonbatched_bias, axis=1)

    if self.gating:
        gate_values = paddle.einsum('nbqc,chv->nbqhv', q_data,
                                    self.gating_w) + self.gating_b
        gate_values_1 = nn.functional.sigmoid(gate_values)
        weighted_avg *= gate_values_1
    
    output = paddle.einsum('nbqhc,hco->nbqo', weighted_avg,
                          self.output_w) + self.output_b
                
  }
    )DOC");
  }
};

class FusedGateAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGateAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("QKVWeight"), "Input", "QKVWeight",
                   "FusedGateAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearWeight"), "Input", "OutLinearWeight",
                   "FusedGateAttentionGrad");

    if (ctx->Attrs().Get<bool>("is_gating")) {
      ctx->SetOutputDim(framework::GradVarName("GateWeight"),
                        ctx->GetInputDim("GateWeight"));
      ctx->SetOutputDim(framework::GradVarName("GateBias"),
                        ctx->GetInputDim("GateBias"));
      ctx->SetOutputDim(framework::GradVarName("GateBiasOut"),
                        ctx->GetInputDim("GateBiasOut"));
      ctx->SetOutputDim(framework::GradVarName("GateOut"),
                        ctx->GetInputDim("GateOut"));
    }

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }

    if (ctx->HasOutput(framework::GradVarName("NonbatchedBias"))) {
      ctx->SetOutputDim(framework::GradVarName("NonbatchedBias"),
                        ctx->GetInputDim("NonbatchedBias"));
    }

    ctx->SetOutputDim(framework::GradVarName("OutLinearBias"),
                      ctx->GetInputDim("OutLinearBias"));

    ctx->SetOutputDim(framework::GradVarName("OutLinearWeight"),
                      ctx->GetInputDim("OutLinearWeight"));

    ctx->SetOutputDim(framework::GradVarName("QKVWeight"),
                      ctx->GetInputDim("QKVWeight"));

    ctx->SetOutputDim(framework::GradVarName("FMHAOut"),
                      ctx->GetInputDim("FMHAOut"));
    ctx->SetOutputDim(framework::GradVarName("QKTVOut"),
                      ctx->GetInputDim("QKTVOut"));
    ctx->SetOutputDim(framework::GradVarName("QKVTransposeOut"),
                      ctx->GetInputDim("QKVTransposeOut"));
    ctx->SetOutputDim(framework::GradVarName("QKOut"),
                      ctx->GetInputDim("QKOut"));
    ctx->SetOutputDim(framework::GradVarName("SoftmaxOut"),
                      ctx->GetInputDim("SoftmaxOut"));

    ctx->SetOutputDim(framework::GradVarName("QKVOut"),
                      ctx->GetInputDim("QKVOut"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedGateAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_gate_attention_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("X", this->Input("X"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    op->SetInput("QKVWeight", this->Input("QKVWeight"));
    op->SetOutput(framework::GradVarName("QKVWeight"),
                  this->InputGrad("QKVWeight"));

    op->SetInput("QKVOut", this->Output("QKVOut"));
    op->SetOutput(framework::GradVarName("QKVOut"), this->OutputGrad("QKVOut"));

    op->SetInput("QKVTransposeOut", this->Output("QKVTransposeOut"));
    op->SetOutput(framework::GradVarName("QKVTransposeOut"),
                  this->OutputGrad("QKVTransposeOut"));

    op->SetInput("QKOut", this->Output("QKOut"));
    op->SetOutput(framework::GradVarName("QKOut"), this->OutputGrad("QKOut"));

    op->SetInput("QKTVOut", this->Output("QKTVOut"));
    op->SetOutput(framework::GradVarName("QKTVOut"),
                  this->OutputGrad("QKTVOut"));

    if (this->HasInput("NonbatchedBias")) {
      op->SetInput("NonbatchedBias", this->Input("NonbatchedBias"));
      op->SetOutput(framework::GradVarName("NonbatchedBias"),
                    this->InputGrad("NonbatchedBias"));
    }

    op->SetInput("SrcMask", this->Input("SrcMask"));

    op->SetInput("SoftmaxOut", this->Output("SoftmaxOut"));
    op->SetOutput(framework::GradVarName("SoftmaxOut"),
                  this->OutputGrad("SoftmaxOut"));

    op->SetInput("FMHAOut", this->Output("FMHAOut"));
    op->SetOutput(framework::GradVarName("FMHAOut"),
                  this->OutputGrad("FMHAOut"));

    op->SetAttrMap(this->Attrs());
    bool is_gating = BOOST_GET_CONST(bool, op->GetAttr("is_gating"));
    if (is_gating) {
      op->SetInput("GateWeight", this->Input("GateWeight"));
      op->SetOutput(framework::GradVarName("GateWeight"),
                    this->InputGrad("GateWeight"));

      op->SetInput("GateBias", this->Input("GateBias"));
      op->SetOutput(framework::GradVarName("GateBias"),
                    this->InputGrad("GateBias"));

      op->SetInput("GateBiasOut", this->Output("GateBiasOut"));
      op->SetOutput(framework::GradVarName("GateBiasOut"),
                    this->OutputGrad("GateBiasOut"));

      op->SetInput("GateOut", this->Output("GateOut"));
      op->SetOutput(framework::GradVarName("GateOut"),
                    this->OutputGrad("GateOut"));
    }

    op->SetInput("OutLinearWeight", this->Input("OutLinearWeight"));
    op->SetOutput(framework::GradVarName("OutLinearWeight"),
                  this->InputGrad("OutLinearWeight"));

    op->SetInput("OutLinearBias", this->Input("OutLinearBias"));
    op->SetOutput(framework::GradVarName("OutLinearBias"),
                  this->InputGrad("OutLinearBias"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_gate_attention, ops::FusedGateAttentionOp,
    ops::FusedGateAttentionOpMaker,
    ops::FusedGateAttentionGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedGateAttentionGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_gate_attention_grad, ops::FusedGateAttentionGradOp);
