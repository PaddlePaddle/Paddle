// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_dot_product_attention_pass.h"
#include <string>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseDotProductAttentionPass::ApplyImpl(ir::Graph *graph) const {
  std::unordered_set<const Node *> nodes_to_remove;
  QKVCache qkv_cache;
  MaskCache mask_cache;
  OutputCache output_cache;

  graph = FuseDotProductAttentionFwd(
      graph, true, &qkv_cache, &mask_cache, &output_cache, &nodes_to_remove);

  graph = FuseDotProductAttentionFwd(
      graph, false, &qkv_cache, &mask_cache, &output_cache, &nodes_to_remove);

  graph = FuseDotProductAttentionBwd(
      graph, true, &qkv_cache, &mask_cache, &output_cache, &nodes_to_remove);

  graph = FuseDotProductAttentionBwd(
      graph, false, &qkv_cache, &mask_cache, &output_cache, &nodes_to_remove);

  GraphSafeRemoveNodes(graph, nodes_to_remove);
}

ir::Graph *FuseDotProductAttentionPass::FuseDotProductAttentionFwd(
    ir::Graph *graph,
    bool with_dropout,
    QKVCache *qkv_cache,
    MaskCache *mask_cache,
    OutputCache *output_cache,
    std::unordered_set<const Node *> *nodes_to_remove) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("dot_product_attention");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::DotProductAttention dot_product_attention_fwd_pattern(
      gpd.mutable_pattern(), "dot_product_attention_fwd");

  dot_product_attention_fwd_pattern(with_dropout);

  int found_pattern_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle  dot_product_attention fuse"
            << " - with_dropout:" << with_dropout;

    QKVMetaData qkv_meta_data;
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_q, attn_q, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_k, attn_k, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_v, attn_v, dot_product_attention_fwd_pattern);
    qkv_meta_data.q_node = attn_q;
    qkv_meta_data.k_node = attn_k;
    qkv_meta_data.v_node = attn_v;
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_q_transpose, attn_q_transpose, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_k_transpose, attn_k_transpose, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_v_transpose, attn_v_transpose, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_q_transpose_out,
                              attn_q_transpose_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_q_transpose_xshape,
                              attn_q_transpose_xshape,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_transpose_out,
                              attn_k_transpose_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_transpose_xshape,
                              attn_k_transpose_xshape,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_transpose_out,
                              attn_v_transpose_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_transpose_xshape,
                              attn_v_transpose_xshape,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_q_scale, attn_q_scale, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_q_scale_out, attn_q_scale_out, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_qk_matmul, attn_qk_matmul, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_qk_matmul_out,
                              attn_qk_matmul_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_mask, attn_mask, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_mask_eleadd, attn_mask_eleadd, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_eleadd_out,
                              attn_mask_eleadd_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_softmax, attn_softmax, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_softmax_out, attn_softmax_out, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul,
                              attn_context_matmul,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul_out,
                              attn_context_matmul_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_transpose, attn_transpose, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_transpose_out,
                              attn_transpose_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_transpose_xshape,
                              attn_transpose_xshape,
                              dot_product_attention_fwd_pattern);

    std::string mha_meta_key = GenerateMetaKey_(attn_q_scale_out->Name(),
                                                attn_k_transpose_out->Name(),
                                                attn_v_transpose_out->Name());

    // To avoid duplicate conversion.
    if (qkv_cache->Exist(mha_meta_key)) {
      return;
    }

    BlockDesc *block = attn_qk_matmul->Op()->Block();
    Attribute op_role = attn_qk_matmul->Op()->GetAttr("op_role");

    // use `attn_mask` as the key to cache the mask_meta_data.
    // In bwd pass, it's name is `attn_mask_eleadd_grad_mask`
    MaskMetaData mask_meta_data;
    auto mask_meta_key = attn_mask->Var()->Name();
    if (!mask_cache->Exist(mask_meta_key)) {
      mask_meta_data.mask_node = attn_mask;
      mask_cache->Insert(mask_meta_key, mask_meta_data);
    } else {
      mask_meta_data = mask_cache->Get(mask_meta_key);
    }

    // softmax_aux_shape = [batch_size, num_heads, seq_len, 1]
    auto softmax_aux_shape = attn_softmax_out->Var()->GetShape();
    softmax_aux_shape[softmax_aux_shape.size() - 1] = 1;

    // create fused_dot_product_attention op
    VarDesc softmax_out_desc(patterns::PDNodeName(scope_name, "softmax_out"));
    softmax_out_desc.SetDataType(proto::VarType::FP32);
    softmax_out_desc.SetShape(softmax_aux_shape);
    auto *softmax_out_node = g->CreateVarNode(&softmax_out_desc);
    VarDesc rng_state_desc(patterns::PDNodeName(scope_name, "rng_state"));
    rng_state_desc.SetDataType(proto::VarType::INT64);
    rng_state_desc.SetShape({2});
    auto *rng_state_node = g->CreateVarNode(&rng_state_desc);
    OpDesc dot_product_attention_fwd_op_desc(block);
    dot_product_attention_fwd_op_desc.SetType("fused_dot_product_attention");
    dot_product_attention_fwd_op_desc.SetInput("q",
                                               {qkv_meta_data.q_node->Name()});
    dot_product_attention_fwd_op_desc.SetInput("k",
                                               {qkv_meta_data.k_node->Name()});
    dot_product_attention_fwd_op_desc.SetInput("v",
                                               {qkv_meta_data.v_node->Name()});
    dot_product_attention_fwd_op_desc.SetInput(
        "bias", {mask_meta_data.mask_node->Name()});
    dot_product_attention_fwd_op_desc.SetInput("cu_seqlen_q", {});
    dot_product_attention_fwd_op_desc.SetInput("cu_seqlen_kv", {});
    dot_product_attention_fwd_op_desc.SetOutput("out",
                                                {attn_transpose_out->Name()});
    dot_product_attention_fwd_op_desc.SetOutput("softmax_out",
                                                {softmax_out_node->Name()});
    dot_product_attention_fwd_op_desc.SetOutput("rng_state",
                                                {rng_state_node->Name()});
    dot_product_attention_fwd_op_desc.SetAttr(
        "scaling_factor",
        PADDLE_GET_CONST(float, attn_q_scale->Op()->GetAttr("scale")));
    dot_product_attention_fwd_op_desc.SetAttr("mask_type_str",
                                              std::string("none"));
    dot_product_attention_fwd_op_desc.SetAttr("bias_type_str",
                                              std::string("post_scale_bias"));
    dot_product_attention_fwd_op_desc.SetAttr("is_training", true);
    dot_product_attention_fwd_op_desc.SetAttr("op_role", op_role);

    if (with_dropout) {
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_dropout, attn_dropout, dot_product_attention_fwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_out,
                                attn_dropout_out,
                                dot_product_attention_fwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_mask,
                                attn_dropout_mask,
                                dot_product_attention_fwd_pattern);
      dot_product_attention_fwd_op_desc.SetAttr(
          "dropout_probability",
          PADDLE_GET_CONST(float, attn_dropout->Op()->GetAttr("dropout_prob")));
      nodes_to_remove->insert(
          {attn_dropout, attn_dropout_out, attn_dropout_mask});
    } else {
      dot_product_attention_fwd_op_desc.SetAttr("dropout_probability", 0.0f);
    }
    auto dot_product_attention_fwd_op_node =
        g->CreateOpNode(&dot_product_attention_fwd_op_desc);

    IR_NODE_LINK_TO(qkv_meta_data.q_node, dot_product_attention_fwd_op_node);
    IR_NODE_LINK_TO(qkv_meta_data.k_node, dot_product_attention_fwd_op_node);
    IR_NODE_LINK_TO(qkv_meta_data.v_node, dot_product_attention_fwd_op_node);
    IR_NODE_LINK_TO(mask_meta_data.mask_node,
                    dot_product_attention_fwd_op_node);
    IR_NODE_LINK_TO(dot_product_attention_fwd_op_node, attn_transpose_out);
    IR_NODE_LINK_TO(dot_product_attention_fwd_op_node, softmax_out_node);
    IR_NODE_LINK_TO(dot_product_attention_fwd_op_node, rng_state_node);

    qkv_cache->Insert(mha_meta_key, qkv_meta_data);
    OutputMetaData output_meta_data;
    output_meta_data.output_node = attn_transpose_out;
    output_meta_data.softmax_output_node = softmax_out_node;
    output_meta_data.rng_state_node = rng_state_node;
    output_cache->Insert(mha_meta_key, output_meta_data);

    nodes_to_remove->insert({attn_q_transpose,        attn_k_transpose,
                             attn_v_transpose,        attn_q_transpose_out,
                             attn_k_transpose_out,    attn_v_transpose_out,
                             attn_q_transpose_xshape, attn_k_transpose_xshape,
                             attn_v_transpose_xshape, attn_q_scale,
                             attn_q_scale_out,        attn_qk_matmul,
                             attn_qk_matmul_out,      attn_mask_eleadd,
                             attn_mask_eleadd_out,    attn_softmax,
                             attn_softmax_out,        attn_context_matmul,
                             attn_context_matmul_out, attn_transpose,
                             attn_transpose_xshape});

    found_pattern_count++;
  };

  gpd(graph, handler);
  AddStatis(found_pattern_count);
  return graph;
}

ir::Graph *FuseDotProductAttentionPass::FuseDotProductAttentionBwd(
    ir::Graph *graph,
    bool with_dropout,
    QKVCache *qkv_cache,
    MaskCache *mask_cache,
    OutputCache *output_cache,
    std::unordered_set<const Node *> *nodes_to_remove) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("dot_product_attention");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::DotProductAttentionGrad dot_product_attention_bwd_pattern(
      gpd.mutable_pattern(), "dot_product_attention_bwd");

  dot_product_attention_bwd_pattern(with_dropout);

  int found_pattern_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle MultiHeadAttnBwd fuse"
            << " - with_dropout:" << with_dropout;

    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dout, attn_dout, dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_transpose_grad,
                              attn_transpose_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_transpose_grad_out,
                              attn_transpose_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul_grad_x,
                              attn_context_matmul_grad_x,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul_grad_y,
                              attn_context_matmul_grad_y,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul_grad,
                              attn_context_matmul_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul_grad_dx,
                              attn_context_matmul_grad_dx,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_context_matmul_grad_dy,
                              attn_context_matmul_grad_dy,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_softmax_grad,
                              attn_softmax_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_softmax_grad_out,
                              attn_softmax_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_eleadd_grad,
                              attn_mask_eleadd_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_eleadd_grad_mask,
                              attn_mask_eleadd_grad_mask,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_eleadd_grad_dx,
                              attn_mask_eleadd_grad_dx,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_qk_matmul_grad_x,
                              attn_qk_matmul_grad_x,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_qk_matmul_grad_y,
                              attn_qk_matmul_grad_y,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_qk_matmul_grad,
                              attn_qk_matmul_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_qk_matmul_grad_dx,
                              attn_qk_matmul_grad_dx,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_qk_matmul_grad_dy,
                              attn_qk_matmul_grad_dy,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_scale_grad, attn_scale_grad, dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_scale_grad_out,
                              attn_scale_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_q_transpose_grad,
                              attn_q_transpose_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_transpose_grad,
                              attn_k_transpose_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_transpose_grad,
                              attn_v_transpose_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dq, attn_dq, dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dk, attn_dk, dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dv, attn_dv, dot_product_attention_bwd_pattern);

    std::string mha_meta_key =
        GenerateMetaKey_(attn_qk_matmul_grad_x->Name(),
                         attn_qk_matmul_grad_y->Name(),
                         attn_context_matmul_grad_y->Name());
    if (!qkv_cache->Exist(mha_meta_key)) {
      return;
    }
    auto mask_meta_data =
        mask_cache->Get(attn_mask_eleadd_grad_mask->Var()->Name());
    auto qkv_meta_data = qkv_cache->Get(mha_meta_key);
    auto output_meta_data = output_cache->Get(mha_meta_key);

    BlockDesc *block = attn_qk_matmul_grad->Op()->Block();
    Attribute op_role = attn_qk_matmul_grad->Op()->GetAttr("op_role");

    // create fused_dot_product_attention_grad op
    OpDesc dot_product_attention_bwd_op_desc(block);
    dot_product_attention_bwd_op_desc.SetType(
        "fused_dot_product_attention_grad");
    dot_product_attention_bwd_op_desc.SetInput("q",
                                               {qkv_meta_data.q_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput("k",
                                               {qkv_meta_data.k_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput("v",
                                               {qkv_meta_data.v_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput(
        "bias", {mask_meta_data.mask_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput("cu_seqlen_q", {});
    dot_product_attention_bwd_op_desc.SetInput("cu_seqlen_kv", {});
    dot_product_attention_bwd_op_desc.SetInput(
        "out", {output_meta_data.output_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput(
        "softmax_out", {output_meta_data.softmax_output_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput(
        "rng_state", {output_meta_data.rng_state_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput(GradVarName("out"),
                                               {attn_dout->Name()});
    dot_product_attention_bwd_op_desc.SetOutput(GradVarName("q"),
                                                {attn_dq->Name()});
    dot_product_attention_bwd_op_desc.SetOutput(GradVarName("k"),
                                                {attn_dk->Name()});
    dot_product_attention_bwd_op_desc.SetOutput(GradVarName("v"),
                                                {attn_dv->Name()});
    dot_product_attention_bwd_op_desc.SetAttr(
        "scaling_factor",
        PADDLE_GET_CONST(float, attn_scale_grad->Op()->GetAttr("scale")));
    dot_product_attention_bwd_op_desc.SetAttr("mask_type_str",
                                              std::string("none"));
    dot_product_attention_bwd_op_desc.SetAttr("bias_type_str",
                                              std::string("post_scale_bias"));
    dot_product_attention_bwd_op_desc.SetAttr("op_role", op_role);
    if (with_dropout) {
      GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_grad,
                                attn_dropout_grad,
                                dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_grad_out,
                                attn_dropout_grad_out,
                                dot_product_attention_bwd_pattern);
      dot_product_attention_bwd_op_desc.SetAttr(
          "dropout_probability",
          PADDLE_GET_CONST(float,
                           attn_dropout_grad->Op()->GetAttr("dropout_prob")));
      nodes_to_remove->insert({attn_dropout_grad, attn_dropout_grad_out});
    } else {
      dot_product_attention_bwd_op_desc.SetAttr("dropout_probability", 0.0f);
    }
    auto dot_product_attention_bwd_op_node =
        g->CreateOpNode(&dot_product_attention_bwd_op_desc);

    IR_NODE_LINK_TO(attn_dout, dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(qkv_meta_data.q_node, dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(qkv_meta_data.k_node, dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(qkv_meta_data.v_node, dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(output_meta_data.output_node,
                    dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(output_meta_data.softmax_output_node,
                    dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(output_meta_data.rng_state_node,
                    dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(mask_meta_data.mask_node,
                    dot_product_attention_bwd_op_node);

    IR_NODE_LINK_TO(dot_product_attention_bwd_op_node, attn_dq);
    IR_NODE_LINK_TO(dot_product_attention_bwd_op_node, attn_dk);
    IR_NODE_LINK_TO(dot_product_attention_bwd_op_node, attn_dv);

    nodes_to_remove->insert({attn_transpose_grad,
                             attn_transpose_grad_out,
                             attn_context_matmul_grad,
                             attn_context_matmul_grad_x,
                             attn_context_matmul_grad_y,
                             attn_context_matmul_grad_dx,
                             attn_context_matmul_grad_dy,
                             attn_softmax_grad,
                             attn_softmax_grad_out,
                             attn_mask_eleadd_grad,
                             attn_mask_eleadd_grad_dx,
                             attn_qk_matmul_grad,
                             attn_qk_matmul_grad_x,
                             attn_qk_matmul_grad_y,
                             attn_qk_matmul_grad_dx,
                             attn_qk_matmul_grad_dy,
                             attn_scale_grad,
                             attn_scale_grad_out,
                             attn_q_transpose_grad,
                             attn_k_transpose_grad,
                             attn_v_transpose_grad});

    qkv_cache->Erase(mha_meta_key);
    output_cache->Erase(mha_meta_key);

    found_pattern_count++;
  };

  gpd(graph, handler);
  AddStatis(found_pattern_count);
  return graph;
}

std::string FuseDotProductAttentionPass::GenerateMetaKey_(
    const std::string &q_name,
    const std::string &k_name,
    const std::string &v_name) const {
  std::string concat_symbol = "|";
  return q_name + concat_symbol + k_name + concat_symbol + v_name;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_dot_product_attention_pass,
              paddle::framework::ir::FuseDotProductAttentionPass);
