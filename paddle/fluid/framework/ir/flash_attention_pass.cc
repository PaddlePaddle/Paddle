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

#include "paddle/fluid/framework/ir/flash_attention_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

PDNode* FlashAttentionPattern::operator()(PDNode* x,
                                          bool scale_before_matmul,
                                          bool is_causal,
                                          bool is_dropout) {
  auto* q_transpose =
      pattern->NewNode(q_transpose_op_repr())->assert_is_op("transpose2");
  auto* q_transpose_xshape = pattern->NewNode(q_transpose_xshape_repr())
                                 ->assert_is_op_output("transpose2", "XShape");
  auto* q_transpose_out = pattern->NewNode(q_transpose_out_repr())
                              ->assert_is_op_output("transpose2");

  auto* k_transpose_in = pattern->NewNode(k_transpose_in_repr())
                             ->assert_is_op_input("transpose2", "Y");
  auto* k_transpose =
      pattern->NewNode(k_transpose_op_repr())->assert_is_op("transpose2");
  auto* k_transpose_xshape = pattern->NewNode(k_transpose_xshape_repr())
                                 ->assert_is_op_output("transpose2", "XShape");
  auto* k_transpose_out = pattern->NewNode(k_transpose_out_repr())
                              ->assert_is_op_output("transpose2");

  auto* v_transpose_in = pattern->NewNode(v_transpose_in_repr())
                             ->assert_is_op_input("transpose2", "Y");
  auto* v_transpose =
      pattern->NewNode(v_transpose_op_repr())->assert_is_op("transpose2");
  auto* v_transpose_xshape = pattern->NewNode(v_transpose_xshape_repr())
                                 ->assert_is_op_output("transpose2", "XShape");
  auto* v_transpose_out = pattern->NewNode(v_transpose_out_repr())
                              ->assert_is_op_output("transpose2");

  auto* qk_matmul =
      pattern->NewNode(qk_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out =
      pattern->NewNode(qk_matmul_out_repr())->assert_is_op_output("matmul_v2");

  auto scale = pattern->NewNode(scale_op_repr())->assert_is_op("scale");
  auto scale_out =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale", "Out");

  auto* qk_softmax =
      pattern->NewNode(qk_softmax_op_repr())->assert_is_op("softmax");
  auto* qk_softmax_out =
      pattern->NewNode(qk_softmax_out_repr())->assert_is_op_output("softmax");

  auto* dropout = pattern->NewNode(dropout_op_repr())->assert_is_op("dropout");
  auto* dropout_mask = pattern->NewNode(dropout_mask_repr())
                           ->assert_is_op_output("dropout", "Mask");
  auto* dropout_out =
      pattern->NewNode(dropout_out_repr())->assert_is_op_output("dropout");

  auto* qkv_matmul =
      pattern->NewNode(qkv_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qkv_matmul_out =
      pattern->NewNode(qkv_matmul_out_repr())->assert_is_op_output("matmul_v2");

  auto* qkv_transpose =
      pattern->NewNode(qkv_transpose_op_repr())->assert_is_op("transpose2");
  auto* qkv_transpose_xshape =
      pattern->NewNode(qkv_transpose_xshape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto* qkv_transpose_out = pattern->NewNode(qkv_transpose_out_repr())
                                ->assert_is_op_output("transpose2");

  q_transpose->LinksFrom({x}).LinksTo({q_transpose_xshape, q_transpose_out});
  k_transpose->LinksFrom({k_transpose_in})
      .LinksTo({k_transpose_xshape, k_transpose_out});
  v_transpose->LinksFrom({v_transpose_in})
      .LinksTo({v_transpose_xshape, v_transpose_out});

  if (is_causal) {
    // TODO(kk)
  }

  if (scale_before_matmul) {
    scale->LinksFrom({k_transpose_out}).LinksTo({scale_out});
    qk_matmul->LinksFrom({q_transpose_out, scale_out}).LinksTo({qk_matmul_out});
    qk_softmax->LinksFrom({qk_matmul_out}).LinksTo({qk_softmax_out});
  } else {
    qk_matmul->LinksFrom({q_transpose_out, k_transpose_out})
        .LinksTo({qk_matmul_out});
    scale->LinksFrom({qk_matmul_out}).LinksTo({scale_out});
    qk_softmax->LinksFrom({scale_out}).LinksTo({qk_softmax_out});
  }

  if (is_dropout) {
    dropout->LinksFrom({qk_softmax_out}).LinksTo({dropout_mask, dropout_out});
    qkv_matmul->LinksFrom({dropout_out, v_transpose_out})
        .LinksTo({qkv_matmul_out});
  } else {
    qkv_matmul->LinksFrom({qk_softmax_out, v_transpose_out})
        .LinksTo({qkv_matmul_out});
  }

  qkv_transpose->LinksFrom({qkv_matmul_out})
      .LinksTo({qkv_transpose_xshape, qkv_transpose_out});

  return qkv_transpose_out;
}

PDNode* FlashAttentionGradPattern::operator()(PDNode* x,
                                              bool scale_before_matmul,
                                              bool is_causal,
                                              bool is_dropout) {
  return x;
}

}  // namespace patterns

void FlashAttentionsPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  graph = FlashAttentionFwd(graph);
  graph = FlashAttentionBwd(graph);
}

ir::Graph* FlashAttentionsPass::FlashAttentionFwd(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(name_scope_, "x"))
                ->AsInput()
                ->assert_is_op_input("transpose2", "X");
  patterns::FlashAttentionPattern fap(gpd.mutable_pattern(),
                                      "flash_attention_pattern");

  fap(x, true, false, true);

  int found_flash_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "handle FlashAttention pass";

    GET_IR_NODE_FROM_SUBGRAPH(q_transpose_op_node, q_transpose_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(k_transpose_op_node, k_transpose_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(v_transpose_op_node, v_transpose_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op_node, scale_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_op_node, qk_matmul_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qk_softmax_op_node, qk_softmax_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(dropout_op_node, dropout_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_op_node, qkv_matmul_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_transpose_op_node, qkv_transpose_op, fap);

    // TODO(kk)

    GraphSafeRemoveNodes(g,
                         {q_transpose_op_node,
                          k_transpose_op_node,
                          v_transpose_op_node,
                          scale_op_node,
                          qk_matmul_op_node,
                          qk_softmax_op_node,
                          dropout_op_node,
                          qkv_matmul_op_node,
                          qkv_transpose_op_node});
    found_flash_attention++;
  };

  gpd(graph, handler);
  AddStatis(found_flash_attention);

  return graph;
}

ir::Graph* FlashAttentionsPass::FlashAttentionBwd(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(name_scope_, "x"))
                ->AsInput()
                ->assert_is_op_input("transpose2", "X@GRAD");
  patterns::FlashAttentionGradPattern fagp(gpd.mutable_pattern(),
                                           "flash_attention_grad_pattern");

  fagp(x, true, false, true);

  int found_flash_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "handle FlashAttention pass";

    GraphSafeRemoveNodes(g, {});

    found_flash_attention++;
  };

  gpd(graph, handler);
  AddStatis(found_flash_attention);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(flash_attention_pass, paddle::framework::ir::FlashAttentionsPass);
