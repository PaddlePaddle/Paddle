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

/*
 * # pattern case 1 : merged_qkv
 *                  q       [scale]
 * | tp  --> split ---> mat -------> softmax --> drop --> mat --> tp |
 *              \      /                               /
 *               \    /                               /
 *                \  / k[scale]                    v /
 *                 \/_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _/
 *
 * | split --> flash_attn |
 *
 * # scale pos
 * 0 no scale
 * 1 scale q
 * 2 scale k
 * 3 scale mat
 */

PDNode* FlashAttentionPattern::operator()(PDNode* x,
                                          int scale_pos,
                                          bool merged_qkv,
                                          bool is_causal,
                                          bool is_dropout) {
  PDNode* q_transpose_out{nullptr};
  PDNode* k_transpose_out{nullptr};
  PDNode* v_transpose_out{nullptr};

  if (merged_qkv) {
    auto* transpose =
        pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");
    auto* transpose_out = pattern->NewNode(transpose_out_repr())
                              ->assert_is_op_output("transpose2", "Out");
    auto* transpose_xshape = pattern->NewNode(transpose_xshape_repr())
                                 ->assert_is_op_output("transpose2", "XShape");

    auto* qkv_split =
        pattern->NewNode(qkv_split_op_repr())->assert_is_op("split");

    q_transpose_out = pattern->NewNode(q_transpose_out_repr())
                          ->assert_is_op_output("split")
                          ->assert_is_op_input("matmul_v2", "X");
    k_transpose_out =
        pattern->NewNode(k_transpose_out_repr())->assert_is_op_output("split");
    v_transpose_out = pattern->NewNode(v_transpose_out_repr())
                          ->assert_is_op_output("split")
                          ->assert_is_op_input("matmul_v2", "Y");

    transpose->LinksFrom({x}).LinksTo({transpose_xshape, transpose_out});
    transpose_out->assert_is_op_input("split", "X");
    qkv_split->LinksFrom({transpose_out})
        .LinksTo({q_transpose_out, k_transpose_out, v_transpose_out});
  } else {
    auto* q_transpose =
        pattern->NewNode(q_transpose_op_repr())->assert_is_op("transpose2");
    auto* q_transpose_xshape =
        pattern->NewNode(q_transpose_xshape_repr())
            ->assert_is_op_output("transpose2", "XShape");

    auto* k_transpose_in = pattern->NewNode(k_transpose_in_repr())
                               ->assert_is_op_input("transpose2", "X");
    auto* k_transpose =
        pattern->NewNode(k_transpose_op_repr())->assert_is_op("transpose2");
    auto* k_transpose_xshape =
        pattern->NewNode(k_transpose_xshape_repr())
            ->assert_is_op_output("transpose2", "XShape");
    auto* v_transpose_in = pattern->NewNode(v_transpose_in_repr())
                               ->assert_is_op_input("transpose2", "X");
    auto* v_transpose =
        pattern->NewNode(v_transpose_op_repr())->assert_is_op("transpose2");
    auto* v_transpose_xshape =
        pattern->NewNode(v_transpose_xshape_repr())
            ->assert_is_op_output("transpose2", "XShape");

    q_transpose_out = pattern->NewNode(q_transpose_out_repr())
                          ->assert_is_op_output("transpose2", "Out")
                          ->assert_is_op_input("matmul_v2", "X");
    k_transpose_out = pattern->NewNode(k_transpose_out_repr())
                          ->assert_is_op_output("transpose2", "Out");
    v_transpose_out = pattern->NewNode(v_transpose_out_repr())
                          ->assert_is_op_output("transpose2", "Out")
                          ->assert_is_op_input("matmul_v2", "Y");

    q_transpose->LinksFrom({x}).LinksTo({q_transpose_xshape, q_transpose_out});
    k_transpose->LinksFrom({k_transpose_in})
        .LinksTo({k_transpose_xshape, k_transpose_out});
    v_transpose->LinksFrom({v_transpose_in})
        .LinksTo({v_transpose_xshape, v_transpose_out});
  }

  auto* qk_matmul =
      pattern->NewNode(qk_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out = pattern->NewNode(qk_matmul_out_repr())
                            ->assert_is_op_output("matmul_v2", "Out");

  auto scale = pattern->NewNode(scale_op_repr())->assert_is_op("scale");
  auto scale_out =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale", "Out");

  PDNode* qk_softmax{nullptr};
  PDNode* qk_softmax_out{nullptr};
  if (is_causal) {
    qk_softmax = pattern->NewNode(qk_softmax_op_repr())
                     ->assert_is_op("fused_softmax_mask_upper_triangle");
    qk_softmax_out =
        pattern->NewNode(qk_softmax_out_repr())
            ->assert_is_op_output("fused_softmax_mask_upper_triangle");
  } else {
    qk_softmax =
        pattern->NewNode(qk_softmax_op_repr())->assert_is_op("softmax");
    qk_softmax_out =
        pattern->NewNode(qk_softmax_out_repr())->assert_is_op_output("softmax");
  }

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

  if (scale_pos == 0) {  // no scale
    qk_matmul->LinksFrom({q_transpose_out, k_transpose_out})
        .LinksTo({qk_matmul_out});
    qk_softmax->LinksFrom({qk_matmul_out}).LinksTo({qk_softmax_out});
  } else if (scale_pos == 1) {  // scale q
    scale->LinksFrom({q_transpose_out}).LinksTo({scale_out});
    qk_matmul->LinksFrom({scale_out, k_transpose_out}).LinksTo({qk_matmul_out});
    qk_softmax->LinksFrom({qk_matmul_out}).LinksTo({qk_softmax_out});
  } else if (scale_pos == 2) {  // scale k
    scale->LinksFrom({k_transpose_out}).LinksTo({scale_out});
    qk_matmul->LinksFrom({q_transpose_out, scale_out}).LinksTo({qk_matmul_out});
    qk_softmax->LinksFrom({qk_matmul_out}).LinksTo({qk_softmax_out});
  } else if (scale_pos == 3) {  // scale mat
    qk_matmul->LinksFrom({q_transpose_out, k_transpose_out})
        .LinksTo({qk_matmul_out});
    scale->LinksFrom({qk_matmul_out}).LinksTo({scale_out});
    qk_softmax->LinksFrom({scale_out}).LinksTo({qk_softmax_out});
  }

  if (is_dropout) {
    auto* dropout =
        pattern->NewNode(dropout_op_repr())->assert_is_op("dropout");
    auto* dropout_mask = pattern->NewNode(dropout_mask_repr())
                             ->assert_is_op_output("dropout", "Mask");
    auto* dropout_out =
        pattern->NewNode(dropout_out_repr())->assert_is_op_output("dropout");

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
                                              int scale_pos,
                                              bool merged_qkv,
                                              bool is_causal,
                                              bool is_dropout) {
  auto* qkv_transpose_grad = pattern->NewNode(qkv_transpose_grad_op_repr())
                                 ->assert_is_op("transpose2_grad");
  auto* qkv_transpose_grad_out =
      pattern->NewNode(qkv_transpose_grad_out_repr())
          ->assert_is_op_output("transpose2_grad", "X@GRAD");
  auto* qkv_transpose_grad_xshape =
      pattern->NewNode(qkv_transpose_grad_xshape_repr())
          ->assert_is_op_input("transpose2_grad", "XShape");
  qkv_transpose_grad->LinksFrom({x, qkv_transpose_grad_xshape})
      .LinksTo({qkv_transpose_grad_out});

  auto* qkv_matmul_grad = pattern->NewNode(qkv_matmul_grad_op_repr())
                              ->assert_is_op("matmul_v2_grad");
  auto* qkv_matmul_grad_x = pattern->NewNode(qkv_matmul_grad_x_repr())
                                ->assert_is_op_input("matmul_v2_grad", "X");
  auto* qkv_matmul_grad_w = pattern->NewNode(qkv_matmul_grad_w_repr())
                                ->assert_is_op_input("matmul_v2_grad", "Y");
  auto* qkv_matmul_grad_x_grad =
      pattern->NewNode(qkv_matmul_grad_x_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "X@GRAD");
  auto* qkv_matmul_grad_w_grad =
      pattern->NewNode(qkv_matmul_grad_w_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "Y@GRAD");

  qkv_transpose_grad_out->assert_is_op_input("matmul_v2_grad", "Out@GRAD");
  qkv_matmul_grad
      ->LinksFrom(
          {qkv_transpose_grad_out, qkv_matmul_grad_x, qkv_matmul_grad_w})
      .LinksTo({qkv_matmul_grad_x_grad, qkv_matmul_grad_w_grad});

  PDNode* qk_softmax_grad{nullptr};
  PDNode* qk_softmax_grad_out{nullptr};
  PDNode* qk_softmax_grad_fwd_out{nullptr};
  if (is_causal) {
    qk_softmax_grad =
        pattern->NewNode(qk_softmax_grad_op_repr())
            ->assert_is_op("fused_softmax_mask_upper_triangle_grad");
    qk_softmax_grad_out =
        pattern->NewNode(qk_softmax_grad_out_repr())
            ->assert_is_op_output("fused_softmax_mask_upper_triangle_grad",
                                  "X@GRAD");
    qk_softmax_grad_fwd_out =
        pattern->NewNode(qk_softmax_grad_fwd_out_repr())
            ->assert_is_op_input("fused_softmax_mask_upper_triangle_grad",
                                 "Softmax");
  } else {
    qk_softmax_grad = pattern->NewNode(qk_softmax_grad_op_repr())
                          ->assert_is_op("softmax_grad");
    qk_softmax_grad_out = pattern->NewNode(qk_softmax_grad_out_repr())
                              ->assert_is_op_output("softmax_grad", "X@GRAD");
    qk_softmax_grad_fwd_out = pattern->NewNode(qk_softmax_grad_fwd_out_repr())
                                  ->assert_is_op_input("softmax_grad", "Out");
  }

  if (is_dropout) {
    auto* dropout_grad =
        pattern->NewNode(dropout_grad_op_repr())->assert_is_op("dropout_grad");
    auto* dropout_grad_mask = pattern->NewNode(dropout_grad_mask_repr())
                                  ->assert_is_op_input("dropout_grad", "Mask");
    auto* dropout_grad_out =
        pattern->NewNode(dropout_grad_out_repr())
            ->assert_is_op_output("dropout_grad", "X@GRAD");
    if (is_causal) {
      dropout_grad_out->assert_is_op_input(
          "fused_softmax_mask_upper_triangle_grad", "Out@GRAD");
    } else {
      dropout_grad_out->assert_is_op_input("softmax_grad", "Out@GRAD");
    }
    qkv_matmul_grad_x_grad->assert_is_op_input("dropout_grad", "Out@GRAD");
    dropout_grad->LinksFrom({dropout_grad_mask, qkv_matmul_grad_x_grad})
        .LinksTo({dropout_grad_out});
    qk_softmax_grad->LinksFrom({dropout_grad_out, qk_softmax_grad_fwd_out})
        .LinksTo({qk_softmax_grad_out});
  } else {
    if (is_causal) {
      qkv_matmul_grad_x_grad->assert_is_op_input(
          "fused_softmax_mask_upper_triangle_grad", "Out@GRAD");
    } else {
      qkv_matmul_grad_x_grad->assert_is_op_input("softmax_grad", "Out@GRAD");
    }
    qk_softmax_grad
        ->LinksFrom({qkv_matmul_grad_x_grad, qk_softmax_grad_fwd_out})
        .LinksTo({qk_softmax_grad_out});
  }

  auto* qk_matmul_grad = pattern->NewNode(qk_matmul_grad_op_repr())
                             ->assert_is_op("matmul_v2_grad");
  auto* qk_matmul_grad_x = pattern->NewNode(qk_matmul_grad_x_repr())
                               ->assert_is_op_input("matmul_v2_grad", "X");
  auto* qk_matmul_grad_w = pattern->NewNode(qk_matmul_grad_w_repr())
                               ->assert_is_op_input("matmul_v2_grad", "Y");
  auto* qk_matmul_grad_x_grad =
      pattern->NewNode(qk_matmul_grad_x_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "X@GRAD");
  auto* qk_matmul_grad_w_grad =
      pattern->NewNode(qk_matmul_grad_w_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "Y@GRAD");

  auto* scale_grad =
      pattern->NewNode(scale_grad_op_repr())->assert_is_op("scale");
  auto* scale_grad_out = pattern->NewNode(scale_grad_out_repr())
                             ->assert_is_op_output("scale", "Out");

  auto* qkv_concat =
      pattern->NewNode(qkv_concat_op_repr())->assert_is_op("concat");
  auto* qkv_concat_out = pattern->NewNode(qkv_concat_out_repr())
                             ->assert_is_op_output("concat", "Out");
  if (scale_pos == 1) {  // scale q
    qk_softmax_grad_out->assert_is_op_input("matmul_v2_grad", "X");
    qk_matmul_grad
        ->LinksFrom({qk_softmax_grad_out, qk_matmul_grad_x, qk_matmul_grad_w})
        .LinksTo({qk_matmul_grad_x_grad, qk_matmul_grad_w_grad});

    qk_matmul_grad_w_grad->assert_is_op_input("scale", "X");
    scale_grad->LinksFrom({qk_matmul_grad_w_grad}).LinksTo({scale_grad_out});
    scale_grad_out->assert_is_op_input("matmul_v2_grad", "Out@GRAD");

    qk_matmul_grad_x_grad->assert_is_op_input("concat");
    scale_grad_out->assert_is_op_input("concat");
    qkv_matmul_grad_w_grad->assert_is_op_input("concat");

    qkv_concat
        ->LinksFrom(
            {qk_matmul_grad_x_grad, scale_grad_out, qkv_matmul_grad_w_grad})
        .LinksTo({qkv_concat_out});
  } else if (scale_pos == 2) {  // scale k
    qk_softmax_grad_out->assert_is_op_input("matmul_v2_grad", "X");
    qk_matmul_grad
        ->LinksFrom({qk_softmax_grad_out, qk_matmul_grad_x, qk_matmul_grad_w})
        .LinksTo({qk_matmul_grad_x_grad, qk_matmul_grad_w_grad});

    qk_matmul_grad_x_grad->assert_is_op_input("scale", "X");
    scale_grad->LinksFrom({qk_matmul_grad_x_grad}).LinksTo({scale_grad_out});
    scale_grad_out->assert_is_op_input("matmul_v2_grad", "Out@GRAD");

    scale_grad_out->assert_is_op_input("concat");
    qk_matmul_grad_w_grad->assert_is_op_input("concat");
    qkv_matmul_grad_w_grad->assert_is_op_input("concat");

    qkv_concat
        ->LinksFrom(
            {scale_grad_out, qk_matmul_grad_w_grad, qkv_matmul_grad_w_grad})
        .LinksTo({qkv_concat_out});
  } else if (scale_pos == 3) {  // scale mat
    qk_softmax_grad_out->assert_is_op_input("scale", "X");
    scale_grad->LinksFrom({qk_softmax_grad_out}).LinksTo({scale_grad_out});
    scale_grad_out->assert_is_op_input("matmul_v2_grad", "Out@GRAD");
    qk_matmul_grad
        ->LinksFrom({scale_grad_out, qk_matmul_grad_x, qk_matmul_grad_w})
        .LinksTo({qk_matmul_grad_x_grad, qk_matmul_grad_w_grad});

    qk_matmul_grad_x_grad->assert_is_op_input("concat");
    qk_matmul_grad_w_grad->assert_is_op_input("concat");
    qkv_matmul_grad_w_grad->assert_is_op_input("concat");

    qkv_concat
        ->LinksFrom({qk_matmul_grad_x_grad,
                     qk_matmul_grad_w_grad,
                     qkv_matmul_grad_w_grad})
        .LinksTo({qkv_concat_out});
  }

  auto* transpose_grad = pattern->NewNode(transpose_grad_op_repr())
                             ->assert_is_op("transpose2_grad");
  auto* transpose_grad_out =
      pattern->NewNode(transpose_grad_out_repr())
          ->assert_is_op_output("transpose2_grad", "X@GRAD");
  auto* transpose_grad_xshape =
      pattern->NewNode(transpose_grad_xshape_repr())
          ->assert_is_op_input("transpose2_grad", "XShape");
  qkv_concat_out->assert_is_op_input("transpose2_grad", "Out@GRAD");

  transpose_grad->LinksFrom({qkv_concat_out, transpose_grad_xshape})
      .LinksTo({transpose_grad_out});
  return transpose_grad_out;
}

}  // namespace patterns

void FlashAttentionsPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  for (auto scale_pos : std::vector<int>({1, 2, 3})) {
    for (auto merged_qkv : std::vector<bool>({true, false})) {
      for (auto is_causal : std::vector<bool>({true, false})) {
        for (auto is_dropout : std::vector<bool>({true, false})) {
          Cache cache_nodes;
          graph = FlashAttentionFwd(graph,
                                    scale_pos,
                                    merged_qkv,
                                    is_causal,
                                    is_dropout,
                                    &cache_nodes);
          graph = FlashAttentionBwd(graph,
                                    scale_pos,
                                    merged_qkv,
                                    is_causal,
                                    is_dropout,
                                    &cache_nodes);
        }
      }
    }
  }
}

ir::Graph* FlashAttentionsPass::FlashAttentionFwd(Graph* graph,
                                                  int scale_pos,
                                                  bool merged_qkv,
                                                  bool is_causal,
                                                  bool is_dropout,
                                                  Cache* cache_nodes) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(name_scope_, "x"))
                ->AsInput()
                ->assert_is_op_input("transpose2", "X");
  patterns::FlashAttentionPattern fap(gpd.mutable_pattern(),
                                      "flash_attention_pattern");

  fap(x, scale_pos, merged_qkv, is_causal, is_dropout);
  VLOG(4) << "FlashAttention pass fwd" << scale_pos << merged_qkv << is_causal
          << is_dropout;

  int found_flash_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "FlashAttention handle pass fwd" << scale_pos << merged_qkv
            << is_causal << is_dropout;

    // only available for fp16 and bf16
    if (subgraph.at(x)->Var()->GetDataType() != proto::VarType::FP16 &&
        subgraph.at(x)->Var()->GetDataType() != proto::VarType::BF16) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(q_transpose_out, q_transpose_out, fap);
    GET_IR_NODE_FROM_SUBGRAPH(k_transpose_out, k_transpose_out, fap);
    GET_IR_NODE_FROM_SUBGRAPH(v_transpose_out, v_transpose_out, fap);

    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, fap);

    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_op, qk_matmul_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_out, qk_matmul_out, fap);

    GET_IR_NODE_FROM_SUBGRAPH(qk_softmax_op, qk_softmax_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qk_softmax_out, qk_softmax_out, fap);

    GET_IR_NODE_FROM_SUBGRAPH(dropout_op, dropout_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, fap);
    GET_IR_NODE_FROM_SUBGRAPH(dropout_mask, dropout_mask, fap);

    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_op, qkv_matmul_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_out, qkv_matmul_out, fap);

    GET_IR_NODE_FROM_SUBGRAPH(qkv_transpose_op, qkv_transpose_op, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_transpose_out, qkv_transpose_out, fap);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_transpose_xshape, qkv_transpose_xshape, fap);

    cache_nodes->insert(std::make_pair("q_transpose_out", q_transpose_out));
    cache_nodes->insert(std::make_pair("k_transpose_out", k_transpose_out));
    cache_nodes->insert(std::make_pair("v_transpose_out", v_transpose_out));
    cache_nodes->insert(std::make_pair("qkv_transpose_out", qkv_transpose_out));

    if (merged_qkv) {
      VLOG(4) << "FlashAttention handle merged_qkv";
      GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, fap);
      GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, fap);

      GET_IR_NODE_FROM_SUBGRAPH(qkv_split_op, qkv_split_op, fap);

      OpDesc split_op_desc;
      split_op_desc.SetType("split");
      split_op_desc.SetInput("X", {subgraph.at(x)->Name()});
      split_op_desc.SetOutput("Out",
                              {q_transpose_out->Name(),
                               k_transpose_out->Name(),
                               v_transpose_out->Name()});
      split_op_desc.SetAttr("axis", 3);
      split_op_desc.SetAttr("num", 3);
      auto split_op = g->CreateOpNode(&split_op_desc);

      IR_NODE_LINK_TO(subgraph.at(x), split_op);
      IR_NODE_LINK_TO(split_op, q_transpose_out);
      IR_NODE_LINK_TO(split_op, k_transpose_out);
      IR_NODE_LINK_TO(split_op, v_transpose_out);

      auto q_size = q_transpose_out->Var()->GetShape();

      VarDesc softmax_lse(patterns::PDNodeName(name_scope_, "softmax_lse"));
      softmax_lse.SetShape({q_size[0], q_size[2], q_size[1]});
      softmax_lse.SetDataType(proto::VarType::FP32);
      softmax_lse.SetStopGradient(static_cast<bool>(true));
      auto* softmax_lse_var = g->CreateVarNode(&softmax_lse);
      VarDesc seed_offset(patterns::PDNodeName(name_scope_, "seed_offset"));
      seed_offset.SetShape({2});
      seed_offset.SetDataType(proto::VarType::INT64);
      seed_offset.SetStopGradient(static_cast<bool>(true));
      auto* seed_offset_var = g->CreateVarNode(&seed_offset);

      OpDesc flashattn_op_desc(transpose_op->Op()->Block());
      flashattn_op_desc.SetType("flash_attn");
      flashattn_op_desc.SetInput("q", {q_transpose_out->Name()});
      flashattn_op_desc.SetInput("k", {k_transpose_out->Name()});
      flashattn_op_desc.SetInput("v", {v_transpose_out->Name()});
      flashattn_op_desc.SetOutput("out", {qkv_transpose_out->Name()});
      if (is_dropout) {
        flashattn_op_desc.SetOutput("softmax", {dropout_out->Name()});
        auto dropout_prob = dropout_op->Op()->GetAttr("dropout_prob");
        flashattn_op_desc.SetAttr("dropout", dropout_prob);
      } else {
        flashattn_op_desc.SetOutput("softmax", {qk_softmax_out->Name()});
        flashattn_op_desc.SetAttr("dropout", 0.0f);
      }
      flashattn_op_desc.SetOutput("softmax_lse", {softmax_lse_var->Name()});
      flashattn_op_desc.SetOutput("seed_offset", {seed_offset_var->Name()});

      cache_nodes->insert(std::make_pair("softmax_lse", softmax_lse_var));
      cache_nodes->insert(std::make_pair("seed_offset", seed_offset_var));

      flashattn_op_desc.SetAttr("causal", is_causal);
      flashattn_op_desc.SetAttr("return_softmax", true);
      auto flashattn_op = g->CreateOpNode(&flashattn_op_desc);

      IR_NODE_LINK_TO(q_transpose_out, flashattn_op);
      IR_NODE_LINK_TO(k_transpose_out, flashattn_op);
      IR_NODE_LINK_TO(v_transpose_out, flashattn_op);
      IR_NODE_LINK_TO(flashattn_op, qkv_transpose_out);
      if (is_dropout) {
        IR_NODE_LINK_TO(flashattn_op, dropout_out);
      } else {
        IR_NODE_LINK_TO(flashattn_op, qk_softmax_out);
      }
      IR_NODE_LINK_TO(flashattn_op, softmax_lse_var);
      IR_NODE_LINK_TO(flashattn_op, seed_offset_var);

      GraphSafeRemoveNodes(g,
                           {transpose_op,
                            qkv_split_op,
                            qk_matmul_op,
                            scale_op,
                            qk_softmax_op,
                            dropout_op,
                            qkv_matmul_op,
                            qkv_transpose_op});
    } else {
      // TODO(kuizhiqing)
    }

    found_flash_attention++;
  };

  gpd(graph, handler);
  AddStatis(found_flash_attention);

  return graph;
}

ir::Graph* FlashAttentionsPass::FlashAttentionBwd(Graph* graph,
                                                  int scale_pos,
                                                  bool merged_qkv,
                                                  bool is_causal,
                                                  bool is_dropout,
                                                  Cache* cache_nodes) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(name_scope_, "x"))
                ->AsInput()
                ->assert_is_op_input("transpose2_grad", "Out@GRAD");
  patterns::FlashAttentionGradPattern fagp(gpd.mutable_pattern(),
                                           "flash_attention_grad_pattern");

  fagp(x, scale_pos, merged_qkv, is_causal, is_dropout);
  VLOG(4) << "FlashAttention pass bwd" << scale_pos << merged_qkv << is_causal
          << is_dropout;

  int found_flash_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "FlashAttention handle pass bwd" << scale_pos << merged_qkv
            << is_causal << is_dropout;

    // only available for fp16 and bf16
    if (subgraph.at(x)->Var()->GetDataType() != proto::VarType::FP16 &&
        subgraph.at(x)->Var()->GetDataType() != proto::VarType::BF16) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_transpose_grad_op, qkv_transpose_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_transpose_grad_out, qkv_transpose_grad_out, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_transpose_grad_xshape, qkv_transpose_grad_xshape, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_grad_op, qkv_matmul_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_grad_x, qkv_matmul_grad_x, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_grad_w, qkv_matmul_grad_w, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_matmul_grad_x_grad, qkv_matmul_grad_x_grad, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_matmul_grad_w_grad, qkv_matmul_grad_w_grad, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(dropout_grad_op, dropout_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(dropout_grad_out, dropout_grad_out, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(dropout_grad_mask, dropout_grad_mask, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(qk_softmax_grad_op, qk_softmax_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_softmax_grad_fwd_out, qk_softmax_grad_fwd_out, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(qk_softmax_grad_out, qk_softmax_grad_out, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(scale_grad_op, scale_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(scale_grad_out, scale_grad_out, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_grad_op, qk_matmul_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_grad_x, qk_matmul_grad_x, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_grad_w, qk_matmul_grad_w, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_matmul_grad_x_grad, qk_matmul_grad_x_grad, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_matmul_grad_w_grad, qk_matmul_grad_w_grad, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(qkv_concat_op, qkv_concat_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_concat_out, qkv_concat_out, fagp);

    GET_IR_NODE_FROM_SUBGRAPH(transpose_grad_op, transpose_grad_op, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_grad_out, transpose_grad_out, fagp);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose_grad_xshape, transpose_grad_xshape, fagp);

    auto* q_transpose_out = cache_nodes->at("q_transpose_out");
    auto* k_transpose_out = cache_nodes->at("k_transpose_out");
    auto* v_transpose_out = cache_nodes->at("v_transpose_out");
    auto* qkv_transpose_out = cache_nodes->at("qkv_transpose_out");
    auto* softmax_lse = cache_nodes->at("softmax_lse");
    auto* seed_offset = cache_nodes->at("seed_offset");

    if (merged_qkv) {
      OpDesc flashattn_op_grad_desc(qkv_transpose_grad_op->Op()->Block());
      flashattn_op_grad_desc.SetType("flash_attn_grad");
      flashattn_op_grad_desc.SetInput("q", {q_transpose_out->Name()});
      flashattn_op_grad_desc.SetInput("k", {k_transpose_out->Name()});
      flashattn_op_grad_desc.SetInput("v", {v_transpose_out->Name()});
      flashattn_op_grad_desc.SetInput("out", {qkv_transpose_out->Name()});
      flashattn_op_grad_desc.SetInput("softmax_lse", {softmax_lse->Name()});
      flashattn_op_grad_desc.SetInput("seed_offset", {seed_offset->Name()});
      flashattn_op_grad_desc.SetInput("out@GRAD", {subgraph.at(x)->Name()});

      if (is_dropout) {
        auto dropout_prob = dropout_grad_op->Op()->GetAttr("dropout_prob");
        flashattn_op_grad_desc.SetAttr("dropout", dropout_prob);
      } else {
        flashattn_op_grad_desc.SetAttr("dropout", 0.0f);
      }

      flashattn_op_grad_desc.SetOutput("q@GRAD",
                                       {qk_matmul_grad_x_grad->Name()});
      flashattn_op_grad_desc.SetOutput("k@GRAD",
                                       {qk_matmul_grad_w_grad->Name()});
      flashattn_op_grad_desc.SetOutput("v@GRAD",
                                       {qkv_matmul_grad_w_grad->Name()});

      flashattn_op_grad_desc.SetAttr("causal", is_causal);
      flashattn_op_grad_desc.SetAttr("return_softmax", false);

      auto flashattn_op_grad = g->CreateOpNode(&flashattn_op_grad_desc);

      IR_NODE_LINK_TO(q_transpose_out, flashattn_op_grad);
      IR_NODE_LINK_TO(k_transpose_out, flashattn_op_grad);
      IR_NODE_LINK_TO(v_transpose_out, flashattn_op_grad);
      IR_NODE_LINK_TO(qkv_transpose_out, flashattn_op_grad);
      IR_NODE_LINK_TO(softmax_lse, flashattn_op_grad);
      IR_NODE_LINK_TO(seed_offset, flashattn_op_grad);
      IR_NODE_LINK_TO(subgraph.at(x), flashattn_op_grad);

      IR_NODE_LINK_TO(flashattn_op_grad, qk_matmul_grad_x_grad);
      IR_NODE_LINK_TO(flashattn_op_grad, qk_matmul_grad_w_grad);
      IR_NODE_LINK_TO(flashattn_op_grad, qkv_matmul_grad_w_grad);

      OpDesc concat_op_desc;
      concat_op_desc.SetType("concat");
      concat_op_desc.SetInput("X",
                              {qk_matmul_grad_x_grad->Name(),
                               qk_matmul_grad_w_grad->Name(),
                               qkv_matmul_grad_w_grad->Name()});
      concat_op_desc.SetOutput("Out", {transpose_grad_out->Name()});
      concat_op_desc.SetAttr("axis", 3);
      concat_op_desc.SetAttr("num", 3);
      auto concat_op = g->CreateOpNode(&concat_op_desc);

      IR_NODE_LINK_TO(qk_matmul_grad_x_grad, concat_op);
      IR_NODE_LINK_TO(qk_matmul_grad_w_grad, concat_op);
      IR_NODE_LINK_TO(qkv_matmul_grad_w_grad, concat_op);
      IR_NODE_LINK_TO(concat_op, transpose_grad_out);

    } else {
      // TODO(kuizhiqing)
    }

    GraphSafeRemoveNodes(g,
                         {qkv_transpose_grad_op,
                          qkv_matmul_grad_op,
                          dropout_grad_op,
                          qk_softmax_grad_op,
                          scale_grad_op,
                          qk_matmul_grad_op,
                          qkv_concat_op,
                          transpose_grad_op});

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
