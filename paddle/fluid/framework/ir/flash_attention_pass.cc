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
    // TODO(kuizhiqing)
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
    // TODO(kuizhiqing)
  };

  gpd(graph, handler);
  AddStatis(found_flash_attention);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(flash_attention_pass, paddle::framework::ir::FlashAttentionsPass);
