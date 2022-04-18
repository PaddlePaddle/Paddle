// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/self_attention_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static void ReplaceOutputVar(Node* op, Node* old_var, Node* new_var) {
  if (op->IsOp() && op->Op()) {
    new_var->inputs.push_back(op);
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_var) {
        op->outputs[i] = new_var;
        op->Op()->RenameOutput(old_var->Name(), new_var->Name());
      }
    }
  }
}

PDNode* SelfAttentionPattern::operator()() {
  // First path with scale
  auto* input_q = pattern->NewNode(input_q_repr())
                      ->AsInput()
                      ->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var =
      pattern->NewNode(reshape2_0_out_repr())->assert_is_op_output("reshape2");
  reshape2_0_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_0_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");
  scale_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul_qk = pattern->NewNode(matmul_qk_repr())->assert_is_op("matmul");
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_op_output("matmul");
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto* eltadd_qk =
      pattern->NewNode(eltadd_qk_repr())->assert_is_op("elementwise_add");
  auto* eltadd_qk_b_var = pattern->NewNode(eltadd_qk_b_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_qk_out_var = pattern->NewNode(eltadd_qk_out_repr())
                                ->assert_is_op_output("elementwise_add");
  eltadd_qk_out_var->AsIntermediate()->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var =
      pattern->NewNode(softmax_qk_out_repr())->assert_is_op_output("softmax");
  softmax_qk_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matmul");
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_op_output("matmul");
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var = pattern->NewNode(reshape2_qkv_out_repr())
                                   ->assert_is_op_output("reshape2");
  reshape2_qkv_out_var->assert_is_op_input("mul");

  // Second path to matmul

  auto* input_k = pattern->NewNode(input_k_repr())
                      ->AsInput()
                      ->assert_is_op_input("reshape2");

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");

  auto* reshape2_1_out_var =
      pattern->NewNode(reshape2_1_out_repr())->assert_is_op_output("reshape2");
  reshape2_1_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_1_out_var->AsIntermediate()->assert_is_op_input(
      "matmul");  // link to matmul qk

  // Third path to matmul

  auto* input_v = pattern->NewNode(input_v_repr())
                      ->AsInput()
                      ->assert_is_op_input("reshape2");

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");

  auto* reshape2_2_out_var =
      pattern->NewNode(reshape2_2_out_repr())->assert_is_op_output("reshape2");
  reshape2_2_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_2_out_var->AsIntermediate()->assert_is_op_input(
      "matmul");  // link to matmul qkv
  // Q path
  reshape2_0->LinksFrom({input_q}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  scale->LinksFrom({transpose2_0_out_var}).LinksTo({scale_out_var});
  // K path
  reshape2_1->LinksFrom({input_k}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({scale_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  reshape2_2->LinksFrom({input_v}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return reshape2_qkv_out_var;
}
}  // namespace patterns

SelfAttentionFusePass::SelfAttentionFusePass() {
  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      // in bias, shape is (B, S, N*H),
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      .AddInput("Y")
      // in bias, shape is (N*H)
      // in biasqk, shape is (B, H, S, S)
      .IsTensor()
      .End()
      // in bias, shape is (B, S, N*H)
      // in biasqk, shape is (B, H, S, S)
      .AddOutput("Out")
      .IsTensor()
      .End()
      // in bias, it equal to 2
      // in biasqk, it equal to -1 or 0
      .AddAttr("axis")
      .IsIntIn({2, -1, 0})
      .End();

  AddOpCompat(OpCompat("reshape2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Shape")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ShapeTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("shape")  // -->(B, S, H, N)  <--(B, S, N*H)
      .IsType<std::vector<int>>()
      .End();

  // -->: (B, S, H, N) -> (B, H, S, N)
  // <--: (B, H, S, N) -> (B, S, H, N)
  AddOpCompat(OpCompat("transpose2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("XShape")
      .IsOptional()
      .IsTensor()
      .End()
      .AddAttr("axis")  // {0, 2, 1, 3}
      .IsType<std::vector<int>>()
      .End();

  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsType<float>()  // copy to new op. so unconstrained.
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.f)
      .End()
      .AddAttr("bias_after_scale")  // bias is 0, so unconstrained.
      .IsType<bool>()
      .End();

  // QK (B, H, S, N)*(B, H, S, N) -> (B, H, S, S)
  // QKV (B, H, S, S)*(B, H, S, N) -> (B, H, S, N)
  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsNumEQ(1.0f)
      .End()
      .AddAttr("transpose_X")
      .IsBoolEQ(false)
      .End()
      .AddAttr("transpose_Y")  // QK(true) QKV(false)
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("softmax"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 3})  // shape is (B, H, S, S), so axis is -1 or 3
      .End();
}

int SelfAttentionFusePass::BuildFusion(Graph* graph,
                                       const std::string& name_scope,
                                       Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::SelfAttentionPattern multihead_pattern(pattern, name_scope);

  multihead_pattern();
  // Create New OpDesc
  auto fuse_creater = [&](
      Node* input_q, Node* input_k, Node* input_v, Node* eltadd_qk_b,
      Node* reshape2, Node* reshape2_qkv_out, Node* scale, Node* scale_out,
      Node* softmax_qk, Node* matmul_qk, Node* reshape2_qkv) {
    auto scale_attr = BOOST_GET_CONST(float, scale->Op()->GetAttr("scale"));

    auto reshape_desc = reshape2->Op();
    int head_number =
        BOOST_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape")).at(2);

    OpDesc attention_op_desc(reshape2->Op()->Block());
    attention_op_desc.SetType("multihead_attention");

    attention_op_desc.SetInput("Q", {input_q->Name()});  // B * S * N * H
    attention_op_desc.SetInput("K", {input_k->Name()});  // B * S * N * H
    attention_op_desc.SetInput("V", {input_v->Name()});  // B * S * N * H
    attention_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    attention_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    attention_op_desc.SetAttr("alpha", scale_attr);
    attention_op_desc.SetAttr("head_number", head_number);

    auto* softmax_qk_op_desc = softmax_qk->Op();
    auto* matmul_qk_op_desc = matmul_qk->Op();
    if (matmul_qk_op_desc->HasAttr("X_scale")) {
      attention_op_desc.SetAttr("qkv2context_plugin_int8", true);
      if (softmax_qk_op_desc->HasAttr("out_threshold")) {
        auto qkv_plugin_scale = BOOST_GET_CONST(
            float, softmax_qk_op_desc->GetAttr("out_threshold"));
        attention_op_desc.SetAttr("dp_probs", qkv_plugin_scale);
      }
    }
    if (reshape2_qkv->Op()->HasAttr("out_threshold")) {
      attention_op_desc.SetAttr("out_threshold",
                                reshape2_qkv->Op()->GetAttr("out_threshold"));
    }
    auto* attention = graph->CreateOpNode(&attention_op_desc);

    IR_NODE_LINK_TO(input_q, attention);
    IR_NODE_LINK_TO(input_k, attention);
    IR_NODE_LINK_TO(input_v, attention);
    IR_NODE_LINK_TO(eltadd_qk_b, attention);

    IR_NODE_LINK_TO(attention, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Op compat check in self_attention_fuse_pass_v2 failed.";
      return;
    }
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input_q, input_q, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input_k, input_k, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input_v, input_v, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out, reshape2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out, transpose2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1_out, reshape2_1_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1_out, transpose2_1_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2_out, reshape2_2_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2_out, transpose2_2_out,
                              multihead_pattern);

    // nodes need be removed

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk_out, softmax_qk_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv_out, matmul_qkv_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv_out, reshape2_qkv_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv, transpose2_qkv,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out, transpose2_qkv_out,
                              multihead_pattern);

    fuse_creater(input_q, input_k, input_v, eltadd_qk_b, reshape2_0,
                 reshape2_qkv_out, scale, scale_out, softmax_qk, matmul_qk,
                 reshape2_qkv);

    std::unordered_set<const Node*> marked_nodes(
        {// input_q,
         // input_k,
         // input_v,
         reshape2_0, reshape2_1, reshape2_2, reshape2_0_out, reshape2_1_out,
         reshape2_2_out, transpose2_0, transpose2_1, transpose2_2,
         transpose2_0_out, transpose2_1_out, transpose2_2_out, matmul_qk,
         matmul_qk_out,
         // eltadd_qk_b,
         eltadd_qk, eltadd_qk_out, softmax_qk, softmax_qk_out, transpose2_qkv,
         transpose2_qkv_out, matmul_qkv, matmul_qkv_out, reshape2_qkv,
         // reshape2_qkv_out
         scale, scale_out});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void SelfAttentionFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the self attention pass, The scope should not be null."));

  int fusion_count = BuildFusion(graph, name_scope_, scope);
  if (fusion_count > 0) {
    graph->Set(kSelfAttentionPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(self_attention_fuse_pass,
              paddle::framework::ir::SelfAttentionFusePass);
REGISTER_PASS_CAPABILITY(self_attention_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .LE("matmul", 1)
            .EQ("softmax", 0));
