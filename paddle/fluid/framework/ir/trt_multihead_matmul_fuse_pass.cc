// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/trt_multihead_matmul_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::framework::ir::patterns {

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

static int BuildFusion(Graph* graph, const std::string& name_scope) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  TrtMultiHeadMatmulPattern multihead_pattern(pattern, name_scope);

  multihead_pattern();
  // Create New OpDesc
  auto fuse_creator = [&](Node* input0,
                          Node* mul0,
                          Node* mul1,
                          Node* mul2,
                          Node* mul0_out,
                          Node* mul1_out,
                          Node* mul2_out,
                          Node* eltadd0_b,
                          Node* eltadd1_b,
                          Node* eltadd2_b,
                          Node* eltadd_qk_b,
                          Node* reshape2,
                          Node* reshape2_qkv_out,
                          Node* scale,
                          Node* scale_out) {
    auto scale_attr = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));
    // auto scale_bias = PADDLE_GET_CONST(float, scale->Op()->GetAttr("bias"));
    // bool after_scale =
    //    PADDLE_GET_CONST(bool, scale->Op()->GetAttr("bias_after_scale"));

    // create multihead
    OpDesc multihead_op_desc(mul0->Op()->Block());

    // create tmp tensor
    VarDesc k_var_desc(*mul1_out->Var());
    k_var_desc.SetName("K" + mul1_out->Name());
    auto* k_var_node = graph->CreateVarNode(&k_var_desc);

    VarDesc q_var_desc(*mul0_out->Var());
    q_var_desc.SetName("Q" + mul0_out->Name());
    auto* q_var_node = graph->CreateVarNode(&q_var_desc);

    VarDesc v_var_desc(*mul2_out->Var());
    v_var_desc.SetName("V" + mul2_out->Name());
    auto* v_var_node = graph->CreateVarNode(&v_var_desc);

    auto reshape_desc = reshape2->Op();
    int head_number =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);

    ReplaceOutputVar(mul0, mul0_out, q_var_node);
    ReplaceOutputVar(mul1, mul1_out, k_var_node);
    ReplaceOutputVar(mul2, mul2_out, v_var_node);

    multihead_op_desc.SetType("multihead_matmul");
    multihead_op_desc.SetInput("Q", {q_var_node->Name()});
    multihead_op_desc.SetInput("K", {k_var_node->Name()});
    multihead_op_desc.SetInput("V", {v_var_node->Name()});

    multihead_op_desc.SetInput("BiasQ", {eltadd0_b->Name()});
    multihead_op_desc.SetInput("BiasK", {eltadd1_b->Name()});
    multihead_op_desc.SetInput("BiasV", {eltadd2_b->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* multihead = graph->CreateOpNode(&multihead_op_desc);
    IR_NODE_LINK_TO(q_var_node, multihead);
    IR_NODE_LINK_TO(k_var_node, multihead);
    IR_NODE_LINK_TO(v_var_node, multihead);

    IR_NODE_LINK_TO(eltadd0_b, multihead);
    IR_NODE_LINK_TO(eltadd1_b, multihead);
    IR_NODE_LINK_TO(eltadd2_b, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, multihead_pattern);

    fuse_creator(input0,
                 mul0,
                 mul1,
                 mul2,
                 mul0_out,
                 mul1_out,
                 mul2_out,
                 eltadd0_b,
                 eltadd1_b,
                 eltadd2_b,
                 eltadd_qk_b,
                 reshape2_0,
                 reshape2_qkv_out,
                 scale,
                 scale_out);

    std::unordered_set<const Node*> marked_nodes(
        {eltadd0,
         eltadd1,
         eltadd2,
         eltadd0_out,
         eltadd1_out,
         eltadd2_out,
         reshape2_0,
         reshape2_1,
         reshape2_2,
         reshape2_0_out,
         reshape2_1_out,
         reshape2_2_out,
         transpose2_0,
         transpose2_1,
         transpose2_2,
         transpose2_0_out,
         transpose2_1_out,
         transpose2_2_out,
         matmul_qk,
         matmul_qk_out,
         eltadd_qk,
         eltadd_qk_out,
         softmax_qk,
         softmax_qk_out,  // dropout_qk, dropout_qk_out,
         transpose2_qkv,
         transpose2_qkv_out,
         matmul_qkv,
         matmul_qkv_out,
         mul0_out,
         mul1_out,
         mul2_out,
         reshape2_qkv,
         scale});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

PDNode* TrtMultiHeadMatmulPattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("matrix_multiply");

  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("matrix_multiply");
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_op_output("matrix_multiply");

  decltype(mul0) eltadd0;
  decltype(mul0) eltadd0_b_var;
  decltype(mul0) eltadd0_out_var;

  mul0_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd0_out_var->AsIntermediate()->assert_is_op_input("reshape2");

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
  scale_out_var->AsIntermediate()->assert_is_op_input("matrix_multiply");

  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_op("matrix_multiply");
  auto* matmul_qk_out_var = pattern->NewNode(matmul_qk_out_repr())
                                ->assert_is_op_output("matrix_multiply");
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
  softmax_qk_out_var->AsIntermediate()->assert_is_op_input("matrix_multiply");

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matrix_multiply");
  auto* matmul_qkv_out_var = pattern->NewNode(matmul_qkv_out_repr())
                                 ->assert_is_op_output("matrix_multiply");
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

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("matrix_multiply");
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_op_output("matrix_multiply");

  decltype(mul1) eltadd1;
  decltype(mul1) eltadd1_b_var;
  decltype(mul1) eltadd1_out_var;

  mul1_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd1_out_var->AsIntermediate()->assert_is_op_input("reshape2");

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
      "matrix_multiply");  // link to matmul qk

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("matrix_multiply");
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_op_output("matrix_multiply");

  decltype(mul2) eltadd2;
  decltype(mul2) eltadd2_b_var;
  decltype(mul2) eltadd2_out_var;

  mul2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd2_out_var->AsIntermediate()->assert_is_op_input("reshape2");

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
      "matrix_multiply");  // link to matmul qkv

  // Q path
  mul0->LinksFrom({input0, mul0_w_var}).LinksTo({mul0_out_var});
  eltadd0->LinksFrom({mul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});

  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  scale->LinksFrom({transpose2_0_out_var}).LinksTo({scale_out_var});
  // K path
  mul1->LinksFrom({input0, mul1_w_var}).LinksTo({mul1_out_var});
  eltadd1->LinksFrom({mul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});
  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({scale_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  mul2->LinksFrom({input0, mul2_w_var}).LinksTo({mul2_out_var});
  eltadd2->LinksFrom({mul2_out_var, eltadd2_b_var}).LinksTo({eltadd2_out_var});
  reshape2_2->LinksFrom({eltadd2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return transpose2_2_out_var;
}

PDNode* TrtMultiHeadMatmulV3Pattern::operator()() {
  auto* input0 = pattern->NewNode(input0_repr());
  input0->assert_is_op_input("matrix_multiply");

  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("matrix_multiply");
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_op_output("matrix_multiply");

  decltype(mul0) eltadd0;
  decltype(mul0) eltadd0_b_var;
  decltype(mul0) eltadd0_out_var;

  mul0_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd0_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var =
      pattern->NewNode(reshape2_0_out_repr())->assert_is_op_output("reshape2");
  reshape2_0_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_0_out_var->AsIntermediate()->assert_is_op_input("matrix_multiply",
                                                             "X");

  auto* matmul_qk =
      pattern->NewNode(matmul_qk_repr())->assert_is_op("matrix_multiply");
  auto* matmul_qk_out_var = pattern->NewNode(matmul_qk_out_repr())
                                ->assert_is_op_output("matrix_multiply");
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
  softmax_qk_out_var->AsIntermediate()->assert_is_op_input("matrix_multiply");

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matrix_multiply");
  auto* matmul_qkv_out_var = pattern->NewNode(matmul_qkv_out_repr())
                                 ->assert_is_op_output("matrix_multiply");
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
  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("matrix_multiply");
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_op_output("matrix_multiply");

  decltype(mul1) eltadd1;
  decltype(mul1) eltadd1_b_var;
  decltype(mul1) eltadd1_out_var;

  mul1_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd1_out_var->AsIntermediate()->assert_is_op_input("reshape2");

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
      "matrix_multiply", "Y");  // link to matmul qk

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("matrix_multiply");
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("matrix_multiply", "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_op_output("matrix_multiply");

  decltype(mul2) eltadd2;
  decltype(mul2) eltadd2_b_var;
  decltype(mul2) eltadd2_out_var;

  mul2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd2_out_var->AsIntermediate()->assert_is_op_input("reshape2");

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
      "matrix_multiply");  // link to matmul qkv

  // Q path
  mul0->LinksFrom({input0, mul0_w_var}).LinksTo({mul0_out_var});
  eltadd0->LinksFrom({mul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});

  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  // K path
  mul1->LinksFrom({input0, mul1_w_var}).LinksTo({mul1_out_var});
  eltadd1->LinksFrom({mul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});
  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({transpose2_0_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  mul2->LinksFrom({input0, mul2_w_var}).LinksTo({mul2_out_var});
  eltadd2->LinksFrom({mul2_out_var, eltadd2_b_var}).LinksTo({eltadd2_out_var});
  reshape2_2->LinksFrom({eltadd2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return transpose2_2_out_var;
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

void TrtMultiHeadMatmulFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count = patterns::BuildFusion(graph, name_scope_);
  AddStatis(fusion_count);
}

TrtMultiHeadMatmulV2FusePass::TrtMultiHeadMatmulV2FusePass() {
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

int TrtMultiHeadMatmulV2FusePass::BuildFusionV2(Graph* graph,
                                                const std::string& name_scope,
                                                Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::TrtMultiHeadMatmulPattern multihead_pattern(pattern, name_scope);

  multihead_pattern();
  // Create New OpDesc
  auto fuse_creator = [&](Node* input0,
                          Node* mul0,
                          Node* mul1,
                          Node* mul2,
                          Node* mul0_out,
                          Node* mul1_out,
                          Node* mul2_out,
                          Node* mul0_w,
                          Node* mul1_w,
                          Node* mul2_w,
                          Node* eltadd0_b,
                          Node* eltadd1_b,
                          Node* eltadd2_b,
                          Node* eltadd_qk_b,
                          Node* reshape2,
                          Node* reshape2_qkv_out,
                          Node* scale,
                          Node* scale_out,
                          Node* softmax_qk,
                          Node* eltadd0,
                          Node* eltadd1,
                          Node* eltadd2,
                          Node* matmul_qk,
                          Node* reshape2_qkv) {
    auto scale_attr = PADDLE_GET_CONST(float, scale->Op()->GetAttr("scale"));

    // mul (B * S * Hidden) x (Hidden * 3 * N * H) = (B * S * 3 * N * H)
    // bias (B * S * 3 * N * H) + bias (3 * N * H)
    // Transpose (B * S * 3 * N * H) -> (3 * B * N * S * H)
    auto* wq_tensor =
        scope->FindVar(mul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wk_tensor =
        scope->FindVar(mul1_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wv_tensor =
        scope->FindVar(mul2_w->Name())->GetMutable<phi::DenseTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<phi::DenseTensor>();

    auto* wq_data = wq_tensor->mutable_data<float>(phi::CPUPlace());
    auto* wk_data = wk_tensor->mutable_data<float>(phi::CPUPlace());
    auto* wv_data = wv_tensor->mutable_data<float>(phi::CPUPlace());
    auto* bq_data = bq_tensor->mutable_data<float>(phi::CPUPlace());
    auto* bk_data = bk_tensor->mutable_data<float>(phi::CPUPlace());
    auto* bv_data = bv_tensor->mutable_data<float>(phi::CPUPlace());

    auto combined_w_dims =
        common::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    auto combined_bias_dims = common::make_ddim({3, bq_tensor->dims()[0]});

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = mul0_w->Var();
    combined_w_desc->SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = eltadd0_b->Var();
    combined_bias_desc->SetShape({3, bq_tensor->dims()[0]});
    combined_bias_desc->SetPersistable(true);

    phi::DenseTensor tmp_combined_w_tensor;
    tmp_combined_w_tensor.Resize(combined_w_dims);
    auto* tmp_combined_w_data =
        tmp_combined_w_tensor.mutable_data<float>(phi::CPUPlace());

    std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
    int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];
    // Combine the three fc weights together.
    for (int i = 0; i < dims_h; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < dims_w; k++) {
          int out_index = i * (3 * dims_w) + j * dims_w + k;
          int in_index = i * dims_w + k;
          tmp_combined_w_data[out_index] = w_vec[j][in_index];
        }
      }
    }

    wq_tensor->Resize(combined_w_dims);
    auto* new_combined_w_data = wq_tensor->mutable_data<float>(phi::CPUPlace());
    memcpy(new_combined_w_data,
           tmp_combined_w_data,
           sizeof(float) * wq_tensor->numel());

    scope->EraseVars({mul1_w->Name(), mul2_w->Name()});
    paddle::memory::Release(phi::CPUPlace());

    phi::DenseTensor tmp_combined_bias_tensor;
    tmp_combined_bias_tensor.Resize(combined_bias_dims);
    auto* tmp_combined_bias_data =
        tmp_combined_bias_tensor.mutable_data<float>(phi::CPUPlace());

    size_t bias_size = bq_tensor->numel();
    memcpy(tmp_combined_bias_data, bq_data, sizeof(float) * bias_size);
    memcpy(
        tmp_combined_bias_data + bias_size, bk_data, sizeof(float) * bias_size);
    memcpy(tmp_combined_bias_data + 2 * bias_size,
           bv_data,
           sizeof(float) * bias_size);

    bq_tensor->Resize(combined_bias_dims);
    auto* new_combined_bias_data =
        bq_tensor->mutable_data<float>(phi::CPUPlace());
    memcpy(new_combined_bias_data,
           tmp_combined_bias_data,
           sizeof(float) * bq_tensor->numel());

    scope->EraseVars({eltadd1_b->Name(), eltadd2_b->Name()});
    paddle::memory::Release(phi::CPUPlace());

    auto reshape_desc = reshape2->Op();
    int head_number =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);

    OpDesc multihead_op_desc(mul0->Op()->Block());
    multihead_op_desc.SetType("multihead_matmul");

    multihead_op_desc.SetInput("Input", {input0->Name()});
    multihead_op_desc.SetInput("W", {mul0_w->Name()});
    multihead_op_desc.SetInput("Bias", {eltadd0_b->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* mul0_op_desc = mul0->Op();

    // all mul op has same input.
    if (mul0_op_desc->HasAttr("Input_scale")) {
      multihead_op_desc.SetAttr("Input_scale",
                                mul0_op_desc->GetAttr("Input_scale"));
    }
    auto* add0_op_desc = eltadd0->Op();
    auto* add1_op_desc = eltadd1->Op();
    auto* add2_op_desc = eltadd2->Op();
    if (add0_op_desc->HasAttr("out_threshold")) {
      auto out_scale0 =
          PADDLE_GET_CONST(float, add0_op_desc->GetAttr("out_threshold"));
      auto out_scale1 =
          PADDLE_GET_CONST(float, add1_op_desc->GetAttr("out_threshold"));
      auto out_scale2 =
          PADDLE_GET_CONST(float, add2_op_desc->GetAttr("out_threshold"));
      auto out_scale_max = std::max(out_scale0, out_scale1);
      out_scale_max = std::max(out_scale_max, out_scale2);
      multihead_op_desc.SetAttr("fc_out_threshold", out_scale_max);
    }

    auto* softmax_qk_op_desc = softmax_qk->Op();
    auto* matmul_qk_op_desc = matmul_qk->Op();
    if (matmul_qk_op_desc->HasAttr("Input_scale")) {
      multihead_op_desc.SetAttr("qkv2context_plugin_int8", true);
      if (softmax_qk_op_desc->HasAttr("out_threshold")) {
        auto qkv_plugin_scale = PADDLE_GET_CONST(
            float, softmax_qk_op_desc->GetAttr("out_threshold"));
        multihead_op_desc.SetAttr("dp_probs", qkv_plugin_scale);
      }
    }
    if (reshape2_qkv->Op()->HasAttr("out_threshold")) {
      multihead_op_desc.SetAttr("out_threshold",
                                reshape2_qkv->Op()->GetAttr("out_threshold"));
    }
    auto* multihead = graph->CreateOpNode(&multihead_op_desc);

    IR_NODE_LINK_TO(input0, multihead);
    IR_NODE_LINK_TO(mul0_w, multihead);
    IR_NODE_LINK_TO(eltadd0_b, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "Op compat check in trt_multihead_matmul_fuse_pass_v2 failed.";
      return;
    }
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, multihead_pattern);

    // If weights or biases in qkv's fc are shared by multiple multihead_matmul
    // patterns, we do not support this kind of fusion, this pass will not take
    // effect.
    bool is_fc_params_shared =
        mul0_w->outputs.size() > 1 || mul1_w->outputs.size() > 1 ||
        mul2_w->outputs.size() > 1 || eltadd0_b->outputs.size() > 1 ||
        eltadd1_b->outputs.size() > 1 || eltadd2_b->outputs.size() > 1;
    if (is_fc_params_shared) {
      return;
    }
    fuse_creator(input0,
                 mul0,
                 mul1,
                 mul2,
                 mul0_out,
                 mul1_out,
                 mul2_out,
                 mul0_w,
                 mul1_w,
                 mul2_w,
                 eltadd0_b,
                 eltadd1_b,
                 eltadd2_b,
                 eltadd_qk_b,
                 reshape2_0,
                 reshape2_qkv_out,
                 scale,
                 scale_out,
                 softmax_qk,
                 eltadd0,
                 eltadd1,
                 eltadd2,
                 matmul_qk,
                 reshape2_qkv);

    std::unordered_set<const Node*> marked_nodes({eltadd0,
                                                  eltadd1,
                                                  eltadd2,
                                                  eltadd1_b,
                                                  eltadd2_b,
                                                  eltadd0_out,
                                                  eltadd1_out,
                                                  eltadd2_out,
                                                  reshape2_0,
                                                  reshape2_1,
                                                  reshape2_2,
                                                  reshape2_0_out,
                                                  reshape2_1_out,
                                                  reshape2_2_out,
                                                  transpose2_0,
                                                  transpose2_1,
                                                  transpose2_2,
                                                  transpose2_0_out,
                                                  transpose2_1_out,
                                                  transpose2_2_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  eltadd_qk,
                                                  eltadd_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  mul0,
                                                  mul1,
                                                  mul2,
                                                  mul0_out,
                                                  mul1_out,
                                                  mul2_out,
                                                  mul1_w,
                                                  mul2_w,
                                                  reshape2_qkv,
                                                  scale});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void TrtMultiHeadMatmulV2FusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::Fatal(
          "During the multiheadMatmul pass, The scope should not be null."));

  int fusion_count = BuildFusionV2(graph, name_scope_, scope);
  if (fusion_count > 0) {
    bool use_varseqlen = Get<bool>("use_varseqlen");
    bool with_interleaved = Get<bool>("with_interleaved");
    std::string pos_id = Get<std::string>("tensorrt_transformer_posid");
    std::string mask_id = Get<std::string>("tensorrt_transformer_maskid");

    if (use_varseqlen && !pos_id.empty() && !mask_id.empty()) {
      if (graph->Has(framework::ir::kEmbEltwiseLayernormPass) ||
          graph->Has(framework::ir::kPrelnEmbEltwiseLayernormPass)) {
        if (with_interleaved) {
          VLOG(3) << "start interleaved_format "
                     "varseqlen_trt_multihead_matmul_fuse_pass_v2";
        } else {
          VLOG(3) << "start varseqlen_trt_multihead_matmul_fuse_pass_v2";
        }
      } else {
        PADDLE_THROW(
            common::errors::Fatal("Use transformer'varseqlen need "
                                  "embedding_eltwise_layernorm_fuse_pass or "
                                  "preln_embedding_eltwise_layernorm_fuse_"
                                  "pass. please use no_varseqlen"));
      }
    } else if (!use_varseqlen && pos_id.empty()) {
      VLOG(3) << "start no_varseqlen_trt_multihead_matmul_fuse_pass";
    } else {
      PADDLE_THROW(
          common::errors::Fatal("Use transformer'varseqlen need config: "
                                "use_varseqlen, set pos_id, set "
                                "mask_id. Or not use varseqlen, do not set "
                                "pos_id. Please "
                                "reconfig"));
    }
    graph->Set(kMultiheadMatmulPass, new bool(true));
  }
  AddStatis(fusion_count);
}

TrtMultiHeadMatmulV3FusePass::TrtMultiHeadMatmulV3FusePass() {
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

int TrtMultiHeadMatmulV3FusePass::BuildFusionV3(Graph* graph,
                                                const std::string& name_scope,
                                                Scope* scope) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  patterns::TrtMultiHeadMatmulV3Pattern multihead_pattern(pattern, name_scope);

  multihead_pattern();
  // Create New OpDesc
  auto fuse_creator = [&](Node* input0,
                          Node* mul0,
                          Node* mul1,
                          Node* mul2,
                          Node* mul0_out,
                          Node* mul1_out,
                          Node* mul2_out,
                          Node* mul0_w,
                          Node* mul1_w,
                          Node* mul2_w,
                          Node* eltadd0_b,
                          Node* eltadd1_b,
                          Node* eltadd2_b,
                          Node* eltadd_qk_b,
                          Node* reshape2,
                          Node* reshape2_qkv_out,
                          Node* matmul_qk) {
    auto scale_attr =
        PADDLE_GET_CONST(float, matmul_qk->Op()->GetAttr("alpha"));

    // mul (B * S * Hidden) x (Hidden * 3 * N * H) = (B * S * 3 * N * H)
    // bias (B * S * 3 * N * H) + bias (3 * N * H)
    // Transpose (B * S * 3 * N * H) -> (3 * B * N * S * H)
    auto* wq_tensor =
        scope->FindVar(mul0_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wk_tensor =
        scope->FindVar(mul1_w->Name())->GetMutable<phi::DenseTensor>();
    auto* wv_tensor =
        scope->FindVar(mul2_w->Name())->GetMutable<phi::DenseTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<phi::DenseTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<phi::DenseTensor>();

    auto* wq_data = wq_tensor->mutable_data<float>(phi::CPUPlace());
    auto* wk_data = wk_tensor->mutable_data<float>(phi::CPUPlace());
    auto* wv_data = wv_tensor->mutable_data<float>(phi::CPUPlace());
    auto* bq_data = bq_tensor->mutable_data<float>(phi::CPUPlace());
    auto* bk_data = bk_tensor->mutable_data<float>(phi::CPUPlace());
    auto* bv_data = bv_tensor->mutable_data<float>(phi::CPUPlace());

    auto combined_w_dims =
        common::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    auto combined_bias_dims = common::make_ddim({3, bq_tensor->dims()[0]});

    // reuse the mul0_w and eltadd_0_b nodes for the combined nodes.
    auto* combined_w_desc = mul0_w->Var();
    combined_w_desc->SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    combined_w_desc->SetPersistable(true);

    auto* combined_bias_desc = eltadd0_b->Var();
    combined_bias_desc->SetShape({3, bq_tensor->dims()[0]});
    combined_bias_desc->SetPersistable(true);

    phi::DenseTensor tmp_combined_w_tensor;
    tmp_combined_w_tensor.Resize(combined_w_dims);
    auto* tmp_combined_w_data =
        tmp_combined_w_tensor.mutable_data<float>(phi::CPUPlace());

    std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
    int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];
    // Combine the three fc weights together.
    for (int i = 0; i < dims_h; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < dims_w; k++) {
          int out_index = i * (3 * dims_w) + j * dims_w + k;
          int in_index = i * dims_w + k;
          tmp_combined_w_data[out_index] = w_vec[j][in_index];
        }
      }
    }

    wq_tensor->Resize(combined_w_dims);
    auto* new_combined_w_data = wq_tensor->mutable_data<float>(phi::CPUPlace());
    memcpy(new_combined_w_data,
           tmp_combined_w_data,
           sizeof(float) * wq_tensor->numel());

    scope->EraseVars({mul1_w->Name(), mul2_w->Name()});
    paddle::memory::Release(phi::CPUPlace());

    phi::DenseTensor tmp_combined_bias_tensor;
    tmp_combined_bias_tensor.Resize(combined_bias_dims);
    auto* tmp_combined_bias_data =
        tmp_combined_bias_tensor.mutable_data<float>(phi::CPUPlace());

    size_t bias_size = bq_tensor->numel();
    memcpy(tmp_combined_bias_data, bq_data, sizeof(float) * bias_size);
    memcpy(
        tmp_combined_bias_data + bias_size, bk_data, sizeof(float) * bias_size);
    memcpy(tmp_combined_bias_data + 2 * bias_size,
           bv_data,
           sizeof(float) * bias_size);

    bq_tensor->Resize(combined_bias_dims);
    auto* new_combined_bias_data =
        bq_tensor->mutable_data<float>(phi::CPUPlace());
    memcpy(new_combined_bias_data,
           tmp_combined_bias_data,
           sizeof(float) * bq_tensor->numel());

    scope->EraseVars({eltadd1_b->Name(), eltadd2_b->Name()});
    paddle::memory::Release(phi::CPUPlace());

    auto reshape_desc = reshape2->Op();
    int head_number =
        PADDLE_GET_CONST(std::vector<int>, reshape_desc->GetAttr("shape"))
            .at(2);

    OpDesc multihead_op_desc(mul0->Op()->Block());
    multihead_op_desc.SetType("multihead_matmul");

    multihead_op_desc.SetInput("Input", {input0->Name()});
    multihead_op_desc.SetInput("W", {mul0_w->Name()});
    multihead_op_desc.SetInput("Bias", {eltadd0_b->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* multihead = graph->CreateOpNode(&multihead_op_desc);

    IR_NODE_LINK_TO(input0, multihead);
    IR_NODE_LINK_TO(mul0_w, multihead);
    IR_NODE_LINK_TO(eltadd0_b, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input0, input0, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_0_out, reshape2_0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_0_out, transpose2_0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_1_out, reshape2_1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_1_out, transpose2_1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_2_out, reshape2_2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_2_out, transpose2_2_out, multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        softmax_qk_out, softmax_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        matmul_qkv_out, matmul_qkv_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        reshape2_qkv_out, reshape2_qkv_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv, transpose2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_qkv_out, transpose2_qkv_out, multihead_pattern);

    // If weights or biases in qkv's fc are shared by multiple multihead_matmul
    // patterns, we do not support this kind of fusion, this pass will not take
    // effect.
    bool is_fc_params_shared =
        mul0_w->outputs.size() > 1 || mul1_w->outputs.size() > 1 ||
        mul2_w->outputs.size() > 1 || eltadd0_b->outputs.size() > 1 ||
        eltadd1_b->outputs.size() > 1 || eltadd2_b->outputs.size() > 1;
    if (is_fc_params_shared) {
      return;
    }
    fuse_creator(input0,
                 mul0,
                 mul1,
                 mul2,
                 mul0_out,
                 mul1_out,
                 mul2_out,
                 mul0_w,
                 mul1_w,
                 mul2_w,
                 eltadd0_b,
                 eltadd1_b,
                 eltadd2_b,
                 eltadd_qk_b,
                 reshape2_0,
                 reshape2_qkv_out,
                 matmul_qk);

    std::unordered_set<const Node*> marked_nodes({eltadd0,
                                                  eltadd1,
                                                  eltadd2,
                                                  eltadd1_b,
                                                  eltadd2_b,
                                                  eltadd0_out,
                                                  eltadd1_out,
                                                  eltadd2_out,
                                                  reshape2_0,
                                                  reshape2_1,
                                                  reshape2_2,
                                                  reshape2_0_out,
                                                  reshape2_1_out,
                                                  reshape2_2_out,
                                                  transpose2_0,
                                                  transpose2_1,
                                                  transpose2_2,
                                                  transpose2_0_out,
                                                  transpose2_1_out,
                                                  transpose2_2_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  eltadd_qk,
                                                  eltadd_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  mul0,
                                                  mul1,
                                                  mul2,
                                                  mul0_out,
                                                  mul1_out,
                                                  mul2_out,
                                                  mul1_w,
                                                  mul2_w,
                                                  reshape2_qkv});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

void TrtMultiHeadMatmulV3FusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      common::errors::Fatal(
          "During the multiheadMatmul pass, The scope should not be null."));

  int fusion_count = BuildFusionV3(graph, name_scope_, scope);
  if (fusion_count > 0) {
    bool use_varseqlen = Get<bool>("use_varseqlen");
    bool with_interleaved = Get<bool>("with_interleaved");
    std::string pos_id = Get<std::string>("tensorrt_transformer_posid");
    std::string mask_id = Get<std::string>("tensorrt_transformer_maskid");

    if (use_varseqlen && !pos_id.empty() && !mask_id.empty()) {
      if (graph->Has(framework::ir::kEmbEltwiseLayernormPass) ||
          graph->Has(framework::ir::kPrelnEmbEltwiseLayernormPass)) {
        if (with_interleaved) {
          VLOG(3) << "start interleaved_format "
                     "varseqlen_trt_multihead_matmul_fuse_pass_v3";
        } else {
          VLOG(3) << "start varseqlen_trt_multihead_matmul_fuse_pass_v3";
        }
      } else {
        PADDLE_THROW(
            common::errors::Fatal("Use transformer'varseqlen need "
                                  "embedding_eltwise_layernorm_fuse_pass or "
                                  "preln_embedding_eltwise_layernorm_fuse_"
                                  "pass. please use no_varseqlen"));
      }
    } else if (!use_varseqlen && pos_id.empty()) {
      VLOG(3) << "start no_varseqlen_trt_multihead_matmul_fuse_pass";
    } else {
      PADDLE_THROW(
          common::errors::Fatal("Use transformer'varseqlen need config: "
                                "use_varseqlen, set pos_id, set "
                                "mask_id. Or not use varseqlen, do not set "
                                "pos_id. Please "
                                "reconfig"));
    }
    graph->Set(kMultiheadMatmulPass, new bool(true));
  }
  AddStatis(fusion_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(trt_multihead_matmul_fuse_pass,
              paddle::framework::ir::TrtMultiHeadMatmulFusePass);

REGISTER_PASS(trt_multihead_matmul_fuse_pass_v2,
              paddle::framework::ir::TrtMultiHeadMatmulV2FusePass);
REGISTER_PASS(trt_multihead_matmul_fuse_pass_v3,
              paddle::framework::ir::TrtMultiHeadMatmulV3FusePass);

REGISTER_PASS_CAPABILITY(trt_multihead_matmul_fuse_pass_v2)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0));

REGISTER_PASS_CAPABILITY(trt_multihead_matmul_fuse_pass_v3)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("reshape2", 0)
            .EQ("transpose2", 0)
            .EQ("scale", 0)
            .EQ("softmax", 0));
