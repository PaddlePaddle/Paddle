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
  SequenceMetaCache seq_meta_cache;
  std::unordered_set<const Node *> nodes_to_remove;
  SoftmaxOutputCache softmax_output_cache;
  QKVCache qkv_cache;

  // self attention forward
  graph = FuseDotProductAttentionFwd(graph,
                                     AttentionType::kSelfAttention,
                                     true,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionFwd(graph,
                                     AttentionType::kSelfAttention,
                                     false,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  // self attention backward
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kSelfAttention,
                                     true,
                                     true,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kSelfAttention,
                                     false,
                                     true,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kSelfAttention,
                                     true,
                                     false,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kSelfAttention,
                                     false,
                                     false,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);

  // cross attention forward
  graph = FuseDotProductAttentionFwd(graph,
                                     AttentionType::kCrossAttention,
                                     true,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionFwd(graph,
                                     AttentionType::kCrossAttention,
                                     false,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  //  cross attention backward
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kCrossAttention,
                                     true,
                                     true,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kCrossAttention,
                                     false,
                                     true,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kCrossAttention,
                                     true,
                                     false,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);
  graph = FuseDotProductAttentionBwd(graph,
                                     AttentionType::kCrossAttention,
                                     false,
                                     false,
                                     &seq_meta_cache,
                                     &softmax_output_cache,
                                     &qkv_cache,
                                     &nodes_to_remove);

  GraphSafeRemoveNodes(graph, nodes_to_remove);
}

ir::Graph *FuseDotProductAttentionPass::FuseDotProductAttentionFwd(
    ir::Graph *graph,
    AttentionType attention_type,
    bool with_dropout,
    SequenceMetaCache *seq_meta_cache,
    SoftmaxOutputCache *softmax_output_cache,
    QKVCache *qkv_cache,
    std::unordered_set<const Node *> *nodes_to_remove) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("dot_product_attention");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::DotProductAttention dot_product_attention_fwd_pattern(
      gpd.mutable_pattern(), "dot_product_attention_fwd");

  const std::string attention_type_name =
      attention_type == AttentionType::kSelfAttention ? "self" : "cross";

  dot_product_attention_fwd_pattern(attention_type_name, with_dropout);

  int found_pattern_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle  dot_product_attention fuse"
            << " - attention_type:" << attention_type_name
            << " - with_dropout:" << with_dropout;

    QKVMetaData qkv_meta_data;
    if (attention_type == AttentionType::kSelfAttention) {
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_qkv, attn_qkv, dot_product_attention_fwd_pattern);
      qkv_meta_data.qkv_node = attn_qkv;
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_q_slice, attn_q_slice, dot_product_attention_fwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_q_slice_out,
                                attn_q_slice_out,
                                dot_product_attention_fwd_pattern);
      nodes_to_remove->insert({attn_q_slice, attn_q_slice_out});
    } else {
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_q, attn_q, dot_product_attention_fwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_kv, attn_kv, dot_product_attention_fwd_pattern);
      qkv_meta_data.q_node = attn_q;
      qkv_meta_data.kv_node = attn_kv;
    }
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_k_slice, attn_k_slice, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_v_slice, attn_v_slice, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_k_slice_out, attn_k_slice_out, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_v_slice_out, attn_v_slice_out, dot_product_attention_fwd_pattern);
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
        attn_mask_cast1, attn_mask_cast1, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_cast1_out,
                              attn_mask_cast1_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_mask_scale1, attn_mask_scale1, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_scale1_out,
                              attn_mask_scale1_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_mask_scale2, attn_mask_scale2, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_scale2_out,
                              attn_mask_scale2_out,
                              dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_mask_cast2, attn_mask_cast2, dot_product_attention_fwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_cast2_out,
                              attn_mask_cast2_out,
                              dot_product_attention_fwd_pattern);
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

    // `attn_mask_cast2_out` can be easily detected in both fwd/bwd pass
    // so we use `attn_mask_cast2_out` instead of `attn_mask` as the key to
    // cache the seq_meta_data.
    // In bwd pass, it's name is `attn_mask_eleadd_grad_mask`
    if (!seq_meta_cache->Exist(attn_mask_cast2_out->Var()->Name())) {
      auto data = InsertActualSeqlenOp_(g, attn_mask, block, op_role);
      seq_meta_cache->Insert(attn_mask_cast2_out->Var()->Name(), data);
    }
    auto seq_meta_data =
        seq_meta_cache->Get(attn_mask_cast2_out->Var()->Name());

    // create fused_dot_product_attention op
    VarDesc softmax_out_desc(patterns::PDNodeName(scope_name, "softmax_out"));
    softmax_out_desc.SetDataType(proto::VarType::FP16);
    softmax_out_desc.SetLoDLevel(attn_softmax_out->Var()->GetLoDLevel());
    auto *softmax_out_node = g->CreateVarNode(&softmax_out_desc);
    OpDesc dot_product_attention_fwd_op_desc(block);
    if (attention_type == AttentionType::kSelfAttention) {
      dot_product_attention_fwd_op_desc.SetType(
          "fused_dot_product_self_attention");
      dot_product_attention_fwd_op_desc.SetInput(
          "QKV", {qkv_meta_data.qkv_node->Name()});
    } else {  // kCrossAttention
      dot_product_attention_fwd_op_desc.SetType(
          "fused_dot_product_cross_attention");
      dot_product_attention_fwd_op_desc.SetInput(
          "Q", {qkv_meta_data.q_node->Name()});
      dot_product_attention_fwd_op_desc.SetInput(
          "KV", {qkv_meta_data.kv_node->Name()});
    }
    dot_product_attention_fwd_op_desc.SetInput(
        "ActualSeqlenQ", {seq_meta_data.q_actual_seqlen_node->Name()});
    dot_product_attention_fwd_op_desc.SetInput(
        "ActualSeqlenKV", {seq_meta_data.kv_actual_seqlen_node->Name()});
    dot_product_attention_fwd_op_desc.SetOutput("Out",
                                                {attn_transpose_out->Name()});
    dot_product_attention_fwd_op_desc.SetOutput("SoftmaxOut",
                                                {softmax_out_node->Name()});
    dot_product_attention_fwd_op_desc.SetAttr(
        "scaling_factor",
        PADDLE_GET_CONST(float, attn_q_scale->Op()->GetAttr("scale")));
    dot_product_attention_fwd_op_desc.SetAttr("is_causal_masking", false);
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
          "attn_dropout_rate",
          PADDLE_GET_CONST(float, attn_dropout->Op()->GetAttr("dropout_prob")));
      auto seed = PADDLE_GET_CONST(int, attn_dropout->Op()->GetAttr("seed"));
      dot_product_attention_fwd_op_desc.SetAttr("attn_dropout_seed", seed);
      nodes_to_remove->insert(
          {attn_dropout, attn_dropout_out, attn_dropout_mask});
    } else {
      dot_product_attention_fwd_op_desc.SetAttr("attn_dropout_seed", 0);
      dot_product_attention_fwd_op_desc.SetAttr("attn_dropout_rate", 0.0f);
    }
    auto dot_product_attention_fwd_op_node =
        g->CreateOpNode(&dot_product_attention_fwd_op_desc);

    if (attention_type == AttentionType::kSelfAttention) {
      IR_NODE_LINK_TO(qkv_meta_data.qkv_node,
                      dot_product_attention_fwd_op_node);
    } else {
      IR_NODE_LINK_TO(qkv_meta_data.q_node, dot_product_attention_fwd_op_node);
      IR_NODE_LINK_TO(qkv_meta_data.kv_node, dot_product_attention_fwd_op_node);
    }
    IR_NODE_LINK_TO(seq_meta_data.q_actual_seqlen_node,
                    dot_product_attention_fwd_op_node);
    IR_NODE_LINK_TO(seq_meta_data.kv_actual_seqlen_node,
                    dot_product_attention_fwd_op_node);
    IR_NODE_LINK_TO(dot_product_attention_fwd_op_node, attn_transpose_out);
    IR_NODE_LINK_TO(dot_product_attention_fwd_op_node, softmax_out_node);

    qkv_cache->Insert(mha_meta_key, qkv_meta_data);
    softmax_output_cache->Insert(mha_meta_key, softmax_out_node);

    nodes_to_remove->insert({attn_k_slice,
                             attn_v_slice,
                             attn_k_slice_out,
                             attn_v_slice_out,
                             attn_q_transpose,
                             attn_k_transpose,
                             attn_v_transpose,
                             attn_q_transpose_out,
                             attn_k_transpose_out,
                             attn_v_transpose_out,
                             attn_q_transpose_xshape,
                             attn_k_transpose_xshape,
                             attn_v_transpose_xshape,
                             attn_q_scale,
                             attn_q_scale_out,
                             attn_qk_matmul,
                             attn_qk_matmul_out,
                             attn_mask_cast1,
                             attn_mask_cast1_out,
                             attn_mask_scale1,
                             attn_mask_scale1_out,
                             attn_mask_scale2,
                             attn_mask_scale2_out,
                             attn_mask_cast2,
                             attn_mask_cast2_out,
                             attn_mask_eleadd,
                             attn_mask_eleadd_out,
                             attn_softmax,
                             attn_softmax_out,
                             attn_context_matmul,
                             attn_context_matmul_out,
                             attn_transpose,
                             attn_transpose_xshape});

    found_pattern_count++;
  };

  gpd(graph, handler);
  AddStatis(found_pattern_count);
  return graph;
}

ir::Graph *FuseDotProductAttentionPass::FuseDotProductAttentionBwd(
    ir::Graph *graph,
    AttentionType attention_type,
    bool with_dropout,
    bool share_attn_mask,
    SequenceMetaCache *seq_meta_cache,
    SoftmaxOutputCache *softmax_output_cache,
    QKVCache *qkv_cache,
    std::unordered_set<const Node *> *nodes_to_remove) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("dot_product_attention");
  FusePassBase::Init(scope_name, graph);

  GraphPatternDetector gpd;
  patterns::DotProductAttentionGrad dot_product_attention_bwd_pattern(
      gpd.mutable_pattern(), "dot_product_attention_bwd");

  const std::string attention_type_name =
      attention_type == AttentionType::kSelfAttention ? "self" : "cross";

  dot_product_attention_bwd_pattern(
      attention_type_name, with_dropout, share_attn_mask);

  int found_pattern_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle MultiHeadAttnBwd fuse"
            << " - attention_type:" << attention_type_name
            << " - with_dropout:" << with_dropout
            << " - share_attn_mask:" << share_attn_mask;

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
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_eleadd_grad_dy,
                              attn_mask_eleadd_grad_dy,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_cast_grad,
                              attn_mask_cast_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_cast_grad_out,
                              attn_mask_cast_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_scale_grad,
                              attn_mask_scale_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_mask_scale_grad_out,
                              attn_mask_scale_grad_out,
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
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_transpose_grad,
                              attn_k_transpose_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_transpose_grad_out,
                              attn_k_transpose_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_transpose_grad,
                              attn_v_transpose_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_transpose_grad_out,
                              attn_v_transpose_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_slice_grad,
                              attn_k_slice_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_k_slice_grad_out,
                              attn_k_slice_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_slice_grad,
                              attn_v_slice_grad,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_v_slice_grad_out,
                              attn_v_slice_grad_out,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_slice_grad_sum,
                              attn_slice_grad_sum,
                              dot_product_attention_bwd_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_q_transpose_grad,
                              attn_q_transpose_grad,
                              dot_product_attention_bwd_pattern);
    Node *dqkv_out = nullptr;
    Node *dq_out = nullptr;
    Node *dkv_out = nullptr;
    if (attention_type == AttentionType::kSelfAttention) {
      GET_IR_NODE_FROM_SUBGRAPH(attn_q_transpose_grad_out,
                                attn_q_transpose_grad_out,
                                dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_q_slice_grad,
                                attn_q_slice_grad,
                                dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_q_slice_grad_out,
                                attn_q_slice_grad_out,
                                dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_dqkv, attn_dqkv, dot_product_attention_bwd_pattern);
      dqkv_out = attn_dqkv;
      nodes_to_remove->insert({attn_q_transpose_grad_out,
                               attn_q_slice_grad,
                               attn_q_slice_grad_out});
    } else {  // attention_type == kCrossAttention
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_dq, attn_dq, dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_dkv, attn_dkv, dot_product_attention_bwd_pattern);
      dq_out = attn_dq;
      dkv_out = attn_dkv;
    }

    if (with_dropout) {
      GET_IR_NODE_FROM_SUBGRAPH(attn_softmax_out,
                                attn_softmax_out,
                                dot_product_attention_bwd_pattern);
    }
    if (share_attn_mask) {
      GET_IR_NODE_FROM_SUBGRAPH(
          attn_mask_sum, attn_mask_sum, dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_mask_sum_out,
                                attn_mask_sum_out,
                                dot_product_attention_bwd_pattern);
      nodes_to_remove->insert({attn_mask_sum, attn_mask_sum_out});
    }

    std::string mha_meta_key =
        GenerateMetaKey_(attn_qk_matmul_grad_x->Name(),
                         attn_qk_matmul_grad_y->Name(),
                         attn_context_matmul_grad_y->Name());
    if (!qkv_cache->Exist(mha_meta_key)) {
      return;
    }
    auto seq_meta_data =
        seq_meta_cache->Get(attn_mask_eleadd_grad_mask->Var()->Name());
    auto qkv_meta_data = qkv_cache->Get(mha_meta_key);
    auto *softmax_out_node = softmax_output_cache->Get(mha_meta_key);

    BlockDesc *block = attn_qk_matmul_grad->Op()->Block();
    Attribute op_role = attn_qk_matmul_grad->Op()->GetAttr("op_role");

    // create fused_dot_product_attention_grad op
    OpDesc dot_product_attention_bwd_op_desc(block);
    if (attention_type == AttentionType::kSelfAttention) {
      dot_product_attention_bwd_op_desc.SetType(
          "fused_dot_product_self_attention_grad");
      dot_product_attention_bwd_op_desc.SetInput(
          "QKV", {qkv_meta_data.qkv_node->Name()});
      dot_product_attention_bwd_op_desc.SetOutput(GradVarName("QKV"),
                                                  {dqkv_out->Name()});
    } else {
      dot_product_attention_bwd_op_desc.SetType(
          "fused_dot_product_cross_attention_grad");
      dot_product_attention_bwd_op_desc.SetInput(
          "Q", {qkv_meta_data.q_node->Name()});
      dot_product_attention_bwd_op_desc.SetInput(
          "KV", {qkv_meta_data.kv_node->Name()});
      dot_product_attention_bwd_op_desc.SetOutput(GradVarName("Q"),
                                                  {dq_out->Name()});
      dot_product_attention_bwd_op_desc.SetOutput(GradVarName("KV"),
                                                  {dkv_out->Name()});
    }
    dot_product_attention_bwd_op_desc.SetInput(GradVarName("Out"),
                                               {attn_dout->Name()});
    dot_product_attention_bwd_op_desc.SetInput("SoftmaxOut",
                                               {softmax_out_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput(
        "ActualSeqlenQ", {seq_meta_data.q_actual_seqlen_node->Name()});
    dot_product_attention_bwd_op_desc.SetInput(
        "ActualSeqlenKV", {seq_meta_data.kv_actual_seqlen_node->Name()});
    dot_product_attention_bwd_op_desc.SetAttr(
        "scaling_factor",
        PADDLE_GET_CONST(float, attn_scale_grad->Op()->GetAttr("scale")));
    // TODO(Shijie Wang): set paddle seed
    dot_product_attention_bwd_op_desc.SetAttr("attn_dropout_seed", 42);
    dot_product_attention_bwd_op_desc.SetAttr("is_causal_masking", false);
    dot_product_attention_bwd_op_desc.SetAttr("op_role", op_role);
    if (with_dropout) {
      GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_grad,
                                attn_dropout_grad,
                                dot_product_attention_bwd_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_grad_out,
                                attn_dropout_grad_out,
                                dot_product_attention_bwd_pattern);
      dot_product_attention_bwd_op_desc.SetAttr(
          "attn_dropout_rate",
          PADDLE_GET_CONST(float,
                           attn_dropout_grad->Op()->GetAttr("dropout_prob")));
      nodes_to_remove->insert({attn_dropout_grad, attn_dropout_grad_out});
    } else {
      dot_product_attention_bwd_op_desc.SetAttr("attn_dropout_rate", 0.0f);
    }
    auto dot_product_attention_bwd_op_node =
        g->CreateOpNode(&dot_product_attention_bwd_op_desc);

    IR_NODE_LINK_TO(attn_dout, dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(softmax_out_node, dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(seq_meta_data.q_actual_seqlen_node,
                    dot_product_attention_bwd_op_node);
    IR_NODE_LINK_TO(seq_meta_data.kv_actual_seqlen_node,
                    dot_product_attention_bwd_op_node);
    if (attention_type == AttentionType::kSelfAttention) {
      IR_NODE_LINK_TO(qkv_meta_data.qkv_node,
                      dot_product_attention_bwd_op_node);
      IR_NODE_LINK_TO(dot_product_attention_bwd_op_node, dqkv_out);

    } else {
      IR_NODE_LINK_TO(qkv_meta_data.q_node, dot_product_attention_bwd_op_node);
      IR_NODE_LINK_TO(qkv_meta_data.kv_node, dot_product_attention_bwd_op_node);
      IR_NODE_LINK_TO(dot_product_attention_bwd_op_node, dq_out);
      IR_NODE_LINK_TO(dot_product_attention_bwd_op_node, dkv_out);
    }

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
                             attn_mask_eleadd_grad_mask,
                             attn_mask_eleadd_grad_dx,
                             attn_mask_eleadd_grad_dy,
                             attn_mask_cast_grad,
                             attn_mask_cast_grad_out,
                             attn_mask_scale_grad,
                             attn_mask_scale_grad_out,
                             attn_qk_matmul_grad,
                             attn_qk_matmul_grad_x,
                             attn_qk_matmul_grad_y,
                             attn_qk_matmul_grad_dx,
                             attn_qk_matmul_grad_dy,
                             attn_scale_grad,
                             attn_scale_grad_out,
                             attn_q_transpose_grad,
                             attn_k_transpose_grad,
                             attn_k_transpose_grad_out,
                             attn_v_transpose_grad,
                             attn_v_transpose_grad_out,
                             attn_k_slice_grad,
                             attn_k_slice_grad_out,
                             attn_v_slice_grad,
                             attn_v_slice_grad_out,
                             attn_slice_grad_sum});

    qkv_cache->Erase(mha_meta_key);
    softmax_output_cache->Erase(mha_meta_key);

    found_pattern_count++;
  };

  gpd(graph, handler);

  AddStatis(found_pattern_count);
  return graph;
}

SequenceMetaData FuseDotProductAttentionPass::InsertActualSeqlenOp_(
    ir::Graph *graph,
    ir::Node *attn_mask,
    BlockDesc *block,
    Attribute op_role) const {
  auto shape = attn_mask->Var()->GetShape();
  SequenceMetaData data;

  VarDesc q_actual_seqlen_desc(
      patterns::PDNodeName("dot_product_attention_fwd", "q_actual_seqlen"));
  q_actual_seqlen_desc.SetDataType(proto::VarType::INT32);
  q_actual_seqlen_desc.SetLoDLevel(attn_mask->Var()->GetLoDLevel());
  auto *q_actual_seqlen_node = graph->CreateVarNode(&q_actual_seqlen_desc);

  VarDesc kv_actual_seqlen_desc(
      patterns::PDNodeName("dot_product_attention_fwd", "kv_actual_seqlen"));
  kv_actual_seqlen_desc.SetDataType(proto::VarType::INT32);
  kv_actual_seqlen_desc.SetLoDLevel(attn_mask->Var()->GetLoDLevel());
  auto *kv_actual_seqlen_node = graph->CreateVarNode(&kv_actual_seqlen_desc);

  if (shape[1] == 1 && shape[2] == 1) {
    // case1: the shape of attn_mask is [b, 1, 1, seqlen]"
    // attn_mask [b, 1, 1, seqlen] -> squeeze ->  k_attn_mask [b, seqlen]
    VLOG(10) << "case1: the shape of attn_mask is [" << shape[0] << ", "
             << shape[1] << ", " << shape[2] << ", " << shape[3] << "]";
    OpDesc attn_mask_squeeze_op_desc(block);
    VarDesc attn_mask_squeeze_out_desc(patterns::PDNodeName(
        "dot_product_attention_fwd", "attn_mask_squeeze_out"));
    attn_mask_squeeze_out_desc.SetDataType(proto::VarType::INT32);
    attn_mask_squeeze_out_desc.SetLoDLevel(attn_mask->Var()->GetLoDLevel());

    auto *attn_mask_squeeze_op_out =
        graph->CreateVarNode(&attn_mask_squeeze_out_desc);

    attn_mask_squeeze_op_desc.SetType("squeeze2");
    attn_mask_squeeze_op_desc.SetInput("X", {attn_mask->Name()});
    attn_mask_squeeze_op_desc.SetOutput("Out",
                                        {attn_mask_squeeze_op_out->Name()});
    attn_mask_squeeze_op_desc.SetAttr("axes", std::vector<int>{1, 2});
    attn_mask_squeeze_op_desc.SetAttr("op_role", op_role);

    auto attn_mask_squeeze_op_node =
        graph->CreateOpNode(&attn_mask_squeeze_op_desc);
    IR_NODE_LINK_TO(attn_mask, attn_mask_squeeze_op_node);
    IR_NODE_LINK_TO(attn_mask_squeeze_op_node, attn_mask_squeeze_op_out);

    // k_attn_mask [b, seqlen] -> ones_like -> q_attn_mask [b, seqlen]
    OpDesc attn_mask_full_like_op_desc(block);
    VarDesc attn_mask_full_like_out_desc(patterns::PDNodeName(
        "dot_product_attention_fwd", "attn_mask_full_like_out"));
    attn_mask_full_like_out_desc.SetDataType(proto::VarType::INT32);
    attn_mask_full_like_out_desc.SetLoDLevel(attn_mask->Var()->GetLoDLevel());
    auto *attn_mask_full_like_op_out =
        graph->CreateVarNode(&attn_mask_full_like_out_desc);

    attn_mask_full_like_op_desc.SetType("fill_any_like");
    attn_mask_full_like_op_desc.SetInput("X",
                                         {attn_mask_squeeze_op_out->Name()});
    attn_mask_full_like_op_desc.SetOutput("Out",
                                          {attn_mask_full_like_op_out->Name()});
    attn_mask_full_like_op_desc.SetAttr("value", 1);
    attn_mask_full_like_op_desc.SetAttr("dtype", 2);
    attn_mask_full_like_op_desc.SetAttr("op_role", op_role);

    auto attn_mask_full_like_op_node =
        graph->CreateOpNode(&attn_mask_full_like_op_desc);
    IR_NODE_LINK_TO(attn_mask_squeeze_op_out, attn_mask_full_like_op_node);
    IR_NODE_LINK_TO(attn_mask_full_like_op_node, attn_mask_full_like_op_out);

    // q_attn_mask [b, seqlen] -> reduce_sum -> q_actual_seqlen [b,]
    OpDesc attn_mask_q_reduce_sum_op_desc(block);

    attn_mask_q_reduce_sum_op_desc.SetType("reduce_sum");
    attn_mask_q_reduce_sum_op_desc.SetInput(
        "X", {attn_mask_full_like_op_out->Name()});
    attn_mask_q_reduce_sum_op_desc.SetOutput("Out",
                                             {q_actual_seqlen_node->Name()});
    attn_mask_q_reduce_sum_op_desc.SetAttr("dim", std::vector<int>{-1});
    attn_mask_q_reduce_sum_op_desc.SetAttr("keep_dim", false);
    attn_mask_q_reduce_sum_op_desc.SetAttr("reduce_all", false);
    attn_mask_q_reduce_sum_op_desc.SetAttr("in_dtype", proto::VarType::INT32);
    attn_mask_q_reduce_sum_op_desc.SetAttr("out_dtype", proto::VarType::INT32);
    attn_mask_q_reduce_sum_op_desc.SetAttr("op_role", op_role);

    auto attn_mask_q_reduce_sum_op_node =
        graph->CreateOpNode(&attn_mask_q_reduce_sum_op_desc);

    IR_NODE_LINK_TO(attn_mask_full_like_op_out, attn_mask_q_reduce_sum_op_node);
    IR_NODE_LINK_TO(attn_mask_q_reduce_sum_op_node, q_actual_seqlen_node);

    // k_attn_mask [b, seqlen] -> reduce_sum -> kv_actual_seqlen [b,]
    OpDesc attn_mask_k_reduce_sum_op_desc(block);

    attn_mask_k_reduce_sum_op_desc.SetType("reduce_sum");
    attn_mask_k_reduce_sum_op_desc.SetInput("X",
                                            {attn_mask_squeeze_op_out->Name()});
    attn_mask_k_reduce_sum_op_desc.SetOutput("Out",
                                             {kv_actual_seqlen_node->Name()});
    attn_mask_k_reduce_sum_op_desc.SetAttr("dim", std::vector<int>{-1});
    attn_mask_k_reduce_sum_op_desc.SetAttr("keep_dim", false);
    attn_mask_k_reduce_sum_op_desc.SetAttr("reduce_all", false);
    attn_mask_k_reduce_sum_op_desc.SetAttr("in_dtype", proto::VarType::INT32);
    attn_mask_k_reduce_sum_op_desc.SetAttr("out_dtype", proto::VarType::INT32);
    attn_mask_k_reduce_sum_op_desc.SetAttr("op_role", op_role);

    auto attn_mask_k_reduce_sum_op_node =
        graph->CreateOpNode(&attn_mask_k_reduce_sum_op_desc);

    IR_NODE_LINK_TO(attn_mask_squeeze_op_out, attn_mask_k_reduce_sum_op_node);
    IR_NODE_LINK_TO(attn_mask_k_reduce_sum_op_node, kv_actual_seqlen_node);
    data.q_actual_seqlen_node = q_actual_seqlen_node;
    data.kv_actual_seqlen_node = kv_actual_seqlen_node;
    return data;
  }
  // case2: the shape of attn_mask is [b, 1, s_q, s_kv]"
  // attn_mask [b, 1, s_q, s_kv] -> slice -> attn_mask_q [b, 1, s_q, 1]
  VLOG(10) << "case2: the shape of attn_mask is [" << shape[0] << ", "
           << shape[1] << ", " << shape[2] << ", " << shape[3] << "]";
  OpDesc attn_mask_q_slice_op_desc(block);
  VarDesc attn_mask_q_slice_op_out_desc(patterns::PDNodeName(
      "dot_product_attention_fwd", "attn_mask_q_slice_out"));
  attn_mask_q_slice_op_out_desc.SetDataType(proto::VarType::INT32);
  attn_mask_q_slice_op_out_desc.SetLoDLevel(attn_mask->Var()->GetLoDLevel());
  auto *attn_mask_q_slice_op_out =
      graph->CreateVarNode(&attn_mask_q_slice_op_out_desc);

  attn_mask_q_slice_op_desc.SetType("slice");
  attn_mask_q_slice_op_desc.SetInput("Input", {attn_mask->Name()});
  attn_mask_q_slice_op_desc.SetOutput("Out",
                                      {attn_mask_q_slice_op_out->Name()});
  attn_mask_q_slice_op_desc.SetAttr("axes", std::vector<int>{3});
  attn_mask_q_slice_op_desc.SetAttr("decrease_axis", std::vector<int>{3});
  attn_mask_q_slice_op_desc.SetAttr("starts", std::vector<int>{0});
  attn_mask_q_slice_op_desc.SetAttr("ends", std::vector<int>{1});
  attn_mask_q_slice_op_desc.SetAttr("op_role", op_role);

  auto attn_mask_q_slice_op_node =
      graph->CreateOpNode(&attn_mask_q_slice_op_desc);
  IR_NODE_LINK_TO(attn_mask, attn_mask_q_slice_op_node);
  IR_NODE_LINK_TO(attn_mask_q_slice_op_node, attn_mask_q_slice_op_out);

  // attn_mask [b, 1, s_q, s_kv] -> slice -> attn_mask_kv [b, 1, 1, s_kv]
  OpDesc attn_mask_kv_slice_op_desc(block);
  VarDesc attn_mask_kv_slice_op_out_desc(patterns::PDNodeName(
      "dot_product_attention_fwd", "attn_mask_kv_slice_out"));
  attn_mask_kv_slice_op_out_desc.SetDataType(proto::VarType::INT32);
  attn_mask_kv_slice_op_out_desc.SetLoDLevel(attn_mask->Var()->GetLoDLevel());
  auto *attn_mask_kv_slice_op_out =
      graph->CreateVarNode(&attn_mask_kv_slice_op_out_desc);

  attn_mask_kv_slice_op_desc.SetType("slice");
  attn_mask_kv_slice_op_desc.SetInput("Input", {attn_mask->Name()});
  attn_mask_kv_slice_op_desc.SetOutput("Out",
                                       {attn_mask_kv_slice_op_out->Name()});
  attn_mask_kv_slice_op_desc.SetAttr("axes", std::vector<int>{2});
  attn_mask_kv_slice_op_desc.SetAttr("decrease_axis", std::vector<int>{2});
  attn_mask_kv_slice_op_desc.SetAttr("starts", std::vector<int>{0});
  attn_mask_kv_slice_op_desc.SetAttr("ends", std::vector<int>{1});
  attn_mask_kv_slice_op_desc.SetAttr("op_role", op_role);

  auto attn_mask_kv_slice_op_node =
      graph->CreateOpNode(&attn_mask_kv_slice_op_desc);
  IR_NODE_LINK_TO(attn_mask, attn_mask_kv_slice_op_node);
  IR_NODE_LINK_TO(attn_mask_kv_slice_op_node, attn_mask_kv_slice_op_out);

  // attn_mask_q [b, 1, s_q, 1] -> reduce_sum -> q_actual_seqlen [b,]
  OpDesc attn_mask_q_reduce_sum_op_desc(block);

  attn_mask_q_reduce_sum_op_desc.SetType("reduce_sum");
  attn_mask_q_reduce_sum_op_desc.SetInput("X",
                                          {attn_mask_q_slice_op_out->Name()});
  attn_mask_q_reduce_sum_op_desc.SetOutput("Out",
                                           {q_actual_seqlen_node->Name()});
  attn_mask_q_reduce_sum_op_desc.SetAttr("dim", std::vector<int>{-1, -2});
  attn_mask_q_reduce_sum_op_desc.SetAttr("keep_dim", false);
  attn_mask_q_reduce_sum_op_desc.SetAttr("reduce_all", false);
  attn_mask_q_reduce_sum_op_desc.SetAttr("in_dtype", proto::VarType::INT32);
  attn_mask_q_reduce_sum_op_desc.SetAttr("out_dtype", proto::VarType::INT32);
  attn_mask_q_reduce_sum_op_desc.SetAttr("op_role", op_role);

  auto attn_mask_q_reduce_sum_op_node =
      graph->CreateOpNode(&attn_mask_q_reduce_sum_op_desc);
  IR_NODE_LINK_TO(attn_mask_q_slice_op_out, attn_mask_q_reduce_sum_op_node);
  IR_NODE_LINK_TO(attn_mask_q_reduce_sum_op_node, q_actual_seqlen_node);

  // attn_mask_kv [b, 1, 1, s_kv] -> reduce_sum -> kv_actual_seqlen [b,]
  OpDesc attn_mask_kv_reduce_sum_op_desc(block);

  attn_mask_kv_reduce_sum_op_desc.SetType("reduce_sum");
  attn_mask_kv_reduce_sum_op_desc.SetInput("X",
                                           {attn_mask_kv_slice_op_out->Name()});
  attn_mask_kv_reduce_sum_op_desc.SetOutput("Out",
                                            {kv_actual_seqlen_node->Name()});
  attn_mask_kv_reduce_sum_op_desc.SetAttr("dim", std::vector<int>{-1, -2});
  attn_mask_kv_reduce_sum_op_desc.SetAttr("keep_dim", false);
  attn_mask_kv_reduce_sum_op_desc.SetAttr("reduce_all", false);
  attn_mask_kv_reduce_sum_op_desc.SetAttr("in_dtype", proto::VarType::INT32);
  attn_mask_kv_reduce_sum_op_desc.SetAttr("out_dtype", proto::VarType::INT32);
  attn_mask_kv_reduce_sum_op_desc.SetAttr("op_role", op_role);

  auto attn_mask_kv_reduce_sum_op_node =
      graph->CreateOpNode(&attn_mask_kv_reduce_sum_op_desc);
  IR_NODE_LINK_TO(attn_mask_kv_slice_op_out, attn_mask_kv_reduce_sum_op_node);
  IR_NODE_LINK_TO(attn_mask_kv_reduce_sum_op_node, kv_actual_seqlen_node);

  data.q_actual_seqlen_node = q_actual_seqlen_node;
  data.kv_actual_seqlen_node = kv_actual_seqlen_node;
  return data;
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
