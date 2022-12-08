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

using Tensor = phi::DenseTensor;
using DDim = framework::DDim;

class FusedGateAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Query"), "Input", "Query", "fused_gate_attention");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearWeight"),
                   "Input",
                   "OutLinearWeight",
                   "fused_gate_attention");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearBias"),
                   "Input",
                   "OutLinearBias",
                   "fused_gate_attention");

    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxOut"),
                   "Output",
                   "SoftmaxOut",
                   "fused_gate_attention");
    OP_INOUT_CHECK(
        ctx->HasOutput("FMHAOut"), "Output", "FMHAOut", "fused_gate_attention");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "fused_gate_attention");

    auto input_q_dims = ctx->GetInputDim("Query");
    int batch_size = input_q_dims[0];
    int seq_len_m = input_q_dims[1];
    int seq_len_r = input_q_dims[2];

    int num_head, m_size, head_dim;
    if (ctx->Attrs().Get<bool>("merge_qkv")) {
      // QKV's input: [batch_size, seq_len_m, seq_len_r, qkv_dim]
      // QKV's weight: [3, num_head, head_dim, qkv_dim]
      OP_INOUT_CHECK(ctx->HasInput("QKVWeight"),
                     "Input",
                     "QKVWeight",
                     "fused_gate_attention");
      OP_INOUT_CHECK(ctx->HasOutput("QKVTransposeOut"),
                     "Output",
                     "QKVTransposeOut",
                     "fused_gate_attention");

      auto qkv_w_dims = ctx->GetInputDim("QKVWeight");

      num_head = qkv_w_dims[1];
      head_dim = qkv_w_dims[2];
      m_size = seq_len_r;

      ctx->SetOutputDim(
          "QKVTransposeOut",
          {3, batch_size, seq_len_m, num_head, seq_len_r, head_dim});
    } else {
      OP_INOUT_CHECK(ctx->HasInput("QueryWeight"),
                     "Input",
                     "QueryWeight",
                     "fused_gate_attention");
      OP_INOUT_CHECK(ctx->HasInput("KeyWeight"),
                     "Input",
                     "KeyWeight",
                     "fused_gate_attention");
      OP_INOUT_CHECK(ctx->HasInput("ValueWeight"),
                     "Input",
                     "ValueWeight",
                     "fused_gate_attention");

      auto input_k_dims = ctx->GetInputDim("Key");
      auto q_w_dims = ctx->GetInputDim("QueryWeight");

      num_head = q_w_dims[1];
      head_dim = q_w_dims[2];
      m_size = input_k_dims[2];

      ctx->SetOutputDim("QueryTransposeOut",
                        {batch_size, seq_len_m, num_head, seq_len_r, head_dim});
      ctx->SetOutputDim("KeyTransposeOut",
                        {batch_size, seq_len_m, num_head, m_size, head_dim});
      ctx->SetOutputDim("ValueTransposeOut",
                        {batch_size, seq_len_m, num_head, m_size, head_dim});
    }

    ctx->SetOutputDim("SoftmaxOut",
                      {batch_size, seq_len_m, num_head, seq_len_r, m_size});
    ctx->SetOutputDim("FMHAOut",
                      {batch_size, seq_len_m, seq_len_r, num_head, head_dim});

    if (ctx->Attrs().Get<bool>("has_gating")) {
      OP_INOUT_CHECK(ctx->HasInput("GateWeight"),
                     "Input",
                     "GateWeight",
                     "fused_gate_attention");
      OP_INOUT_CHECK(ctx->HasInput("GateBias"),
                     "Input",
                     "GateBias",
                     "fused_gate_attention");
      ctx->SetOutputDim("GateOut",
                        {batch_size, seq_len_m, seq_len_r, num_head, head_dim});
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("Query"));
  }
};

class FusedGateAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Query", "The query tensor.");
    AddInput("Key", "The key tensor.").AsDispensable();
    AddInput("QueryWeight", "(optional) The query weight tensor.")
        .AsDispensable();
    AddInput("KeyWeight", "(optional)  The key weight tensor.").AsDispensable();
    AddInput("ValueWeight", "(optional)  The value weight tensor.")
        .AsDispensable();
    AddInput("QKVWeight", "(optional)  The qkv weight tensor.").AsDispensable();
    AddInput("NonbatchedBias", "(optional) The nonbatchedBias tensor.")
        .AsDispensable();
    AddInput("SrcMask", "The attention mask tensor in fmha.");
    AddInput("GateWeight", "(optional) The gate weight tensor.")
        .AsDispensable();
    AddInput("GateBias", "(optional) The gate bias tensor.").AsDispensable();
    AddInput("OutLinearWeight", "The out_linear weight tensor.");
    AddInput("OutLinearBias", "The out_linear bias tensor.");
    AddOutput("QueryTransposeOut", "The transposed result of query matmul.")
        .AsIntermediate()
        .AsDispensable();
    AddOutput("KeyTransposeOut", "The transposed result of key matmul.")
        .AsIntermediate()
        .AsDispensable();
    AddOutput("ValueTransposeOut", "The transposed result of value matmul.")
        .AsIntermediate()
        .AsDispensable();
    AddOutput("QKVTransposeOut", "The transposed result of merged QKV matmul.")
        .AsIntermediate()
        .AsDispensable();
    AddOutput("SoftmaxOut", "Result in fmha.").AsIntermediate();
    AddOutput("FMHAOut", "Result in fmha.").AsIntermediate();
    AddOutput("GateOut", "Result of the gating module.")
        .AsIntermediate()
        .AsDispensable();
    AddOutput("Out", "Result after attention.");
    AddAttr<bool>("has_gating",
                  "if true, the attention op uses gate architecure, "
                  "[default true].")
        .SetDefault(true);
    AddAttr<bool>("merge_qkv",
                  "if true, calculation with merged qkv, "
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

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Query"), "Input", "Query", "fused_gate_attention_grad");
    if (ctx->HasOutput(framework::GradVarName("Query"))) {
      ctx->SetOutputDim(framework::GradVarName("Query"),
                        ctx->GetInputDim("Query"));
    }
    if (ctx->HasOutput(framework::GradVarName("Key"))) {
      ctx->SetOutputDim(framework::GradVarName("Key"), ctx->GetInputDim("Key"));
    }

    if (ctx->Attrs().Get<bool>("merge_qkv")) {
      OP_INOUT_CHECK(ctx->HasInput("QKVWeight"),
                     "Input",
                     "QKVWeight",
                     "fused_gate_attention_arad");
      ctx->SetOutputDim(framework::GradVarName("QKVWeight"),
                        ctx->GetInputDim("QKVWeight"));
    } else {
      OP_INOUT_CHECK(ctx->HasInput("QueryWeight"),
                     "Input",
                     "QueryWeight",
                     "fused_aate_attention_arad");
      OP_INOUT_CHECK(ctx->HasInput("KeyWeight"),
                     "Input",
                     "KeyWeight",
                     "fused_aate_attention_arad");
      OP_INOUT_CHECK(ctx->HasInput("ValueWeight"),
                     "Input",
                     "ValueWeight",
                     "fused_aate_attention_arad");

      for (auto& name : {"QueryWeight", "KeyWeight", "ValueWeight"}) {
        ctx->SetOutputDim(framework::GradVarName(name), ctx->GetInputDim(name));
      }
    }

    OP_INOUT_CHECK(ctx->HasInput("OutLinearWeight"),
                   "Input",
                   "OutLinearWeight",
                   "fused_aate_attention_arad");

    if (ctx->Attrs().Get<bool>("has_gating")) {
      for (auto& name : {"GateWeight", "GateBias"}) {
        ctx->SetOutputDim(framework::GradVarName(name), ctx->GetInputDim(name));
      }
    }

    if (ctx->HasOutput(framework::GradVarName("NonbatchedBias"))) {
      ctx->SetOutputDim(framework::GradVarName("NonbatchedBias"),
                        ctx->GetInputDim("NonbatchedBias"));
    }

    ctx->SetOutputDim(framework::GradVarName("OutLinearWeight"),
                      ctx->GetInputDim("OutLinearWeight"));
    ctx->SetOutputDim(framework::GradVarName("OutLinearBias"),
                      ctx->GetInputDim("OutLinearBias"));
  }
};

template <typename T>
class FusedGateAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_gate_attention_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetInput("Query", this->Input("Query"));
    op->SetOutput(framework::GradVarName("Query"), this->InputGrad("Query"));

    op->SetAttrMap(this->Attrs());
    bool merge_qkv = PADDLE_GET_CONST(bool, op->GetAttr("merge_qkv"));
    if (merge_qkv) {
      op->SetInput("QKVWeight", this->Input("QKVWeight"));
      op->SetOutput(framework::GradVarName("QKVWeight"),
                    this->InputGrad("QKVWeight"));
      op->SetInput("QKVTransposeOut", this->Output("QKVTransposeOut"));
    } else {
      op->SetInput("Key", this->Input("Key"));
      op->SetOutput(framework::GradVarName("Key"), this->InputGrad("Key"));

      for (auto& name : {"QueryWeight", "KeyWeight", "ValueWeight"}) {
        op->SetInput(name, this->Input(name));
        op->SetOutput(framework::GradVarName(name), this->InputGrad(name));
      }

      for (auto& name :
           {"QueryTransposeOut", "KeyTransposeOut", "ValueTransposeOut"}) {
        op->SetInput(name, this->Output(name));
      }
    }

    op->SetInput("FMHAOut", this->Output("FMHAOut"));

    if (this->HasInput("NonbatchedBias")) {
      op->SetInput("NonbatchedBias", this->Input("NonbatchedBias"));
      op->SetOutput(framework::GradVarName("NonbatchedBias"),
                    this->InputGrad("NonbatchedBias"));
    }

    op->SetInput("SoftmaxOut", this->Output("SoftmaxOut"));

    bool has_gating = PADDLE_GET_CONST(bool, op->GetAttr("has_gating"));
    if (has_gating) {
      op->SetInput("GateWeight", this->Input("GateWeight"));
      op->SetOutput(framework::GradVarName("GateWeight"),
                    this->InputGrad("GateWeight"));

      op->SetInput("GateBias", this->Input("GateBias"));
      op->SetOutput(framework::GradVarName("GateBias"),
                    this->InputGrad("GateBias"));

      op->SetInput("GateOut", this->Output("GateOut"));
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
    fused_gate_attention,
    ops::FusedGateAttentionOp,
    ops::FusedGateAttentionOpMaker,
    ops::FusedGateAttentionGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedGateAttentionGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_gate_attention_grad, ops::FusedGateAttentionGradOp);
