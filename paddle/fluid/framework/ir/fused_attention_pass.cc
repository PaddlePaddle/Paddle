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

#include "paddle/fluid/framework/ir/fused_attention_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

PDNode* FusedAttentionPattern::operator()(PDNode* x,
                                          bool pre_layer_norm,
                                          bool post_layer_norm,
                                          bool has_attn_mask,
                                          bool do_dropout,
                                          bool add_residual) {
  // pre layer norm pattern
  PDNode* pre_layer_norm_out_node{nullptr};
  if (pre_layer_norm) {
    auto* pre_layer_norm_node =
        pattern->NewNode(pre_layer_norm_op_repr())->assert_is_op("layer_norm");
    auto* pre_layer_norm_scale_node =
        pattern->NewNode(pre_layer_norm_scale_repr())
            ->assert_is_op_input("layer_norm", "Scale");
    auto* pre_layer_norm_bias_node =
        pattern->NewNode(pre_layer_norm_bias_repr())
            ->assert_is_op_input("layer_norm", "Bias");
    pre_layer_norm_out_node = pattern->NewNode(pre_layer_norm_out_repr())
                                  ->assert_is_op_output("layer_norm", "Y");
    auto* pre_layer_norm_mean_node =
        pattern->NewNode(pre_layer_norm_mean_repr())
            ->assert_is_op_output("layer_norm", "Mean");
    auto* pre_layer_norm_variance_node =
        pattern->NewNode(pre_layer_norm_variance_repr())
            ->assert_is_op_output("layer_norm", "Variance");
    pre_layer_norm_node
        ->LinksFrom({x, pre_layer_norm_scale_node, pre_layer_norm_bias_node})
        .LinksTo({pre_layer_norm_out_node,
                  pre_layer_norm_mean_node,
                  pre_layer_norm_variance_node});
  }

  // fuse qkv pattern
  auto* fuse_qkv_matmul_node =
      pattern->NewNode(fuse_qkv_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* fuse_qkv_matmul_w_node = pattern->NewNode(fuse_qkv_matmul_w_repr())
                                     ->assert_is_op_input("matmul_v2", "Y");
  auto* fuse_qkv_matmul_out_node =
      pattern->NewNode(fuse_qkv_matmul_out_repr())
          ->assert_is_op_output("matmul_v2", "Out");
  if (pre_layer_norm) {
    pre_layer_norm_out_node->AsIntermediate()->assert_is_op_input("matmul_v2",
                                                                  "X");
    fuse_qkv_matmul_node
        ->LinksFrom({pre_layer_norm_out_node, fuse_qkv_matmul_w_node})
        .LinksTo({fuse_qkv_matmul_out_node});
  } else {
    fuse_qkv_matmul_node->LinksFrom({x, fuse_qkv_matmul_w_node})
        .LinksTo({fuse_qkv_matmul_out_node});
  }

  auto* fuse_qkv_ele_add_node = pattern->NewNode(fuse_qkv_ele_add_op_repr())
                                    ->assert_is_op("elementwise_add");
  auto* fuse_qkv_ele_add_bias_node =
      pattern->NewNode(fuse_qkv_ele_add_bias_repr())
          ->assert_is_op_input("elementwise_add", "Y");
  auto* fuse_qkv_ele_add_out_node =
      pattern->NewNode(fuse_qkv_ele_add_out_repr())
          ->assert_is_op_output("elementwise_add", "Out");
  fuse_qkv_matmul_out_node->AsIntermediate()->assert_is_op_input(
      "elementwise_add", "X");
  fuse_qkv_ele_add_node
      ->LinksFrom({fuse_qkv_matmul_out_node, fuse_qkv_ele_add_bias_node})
      .LinksTo({fuse_qkv_ele_add_out_node});

  auto* fuse_qkv_reshape_node =
      pattern->NewNode(fuse_qkv_reshape_op_repr())->assert_is_op("reshape2");
  auto* fuse_qkv_reshape_x_shape_node =
      pattern->NewNode(fuse_qkv_reshape_x_shape_repr())
          ->assert_is_op_output("reshape2", "XShape");
  auto* fuse_qkv_reshape_out_node =
      pattern->NewNode(fuse_qkv_reshape_out_repr())
          ->assert_is_op_output("reshape2", "Out");
  fuse_qkv_ele_add_out_node->AsIntermediate()->assert_is_op_input("reshape2",
                                                                  "X");
  fuse_qkv_reshape_node->LinksFrom({fuse_qkv_ele_add_out_node})
      .LinksTo({fuse_qkv_reshape_x_shape_node, fuse_qkv_reshape_out_node});

  auto* fuse_qkv_transpose_node = pattern->NewNode(fuse_qkv_transpose_op_repr())
                                      ->assert_is_op("transpose2");
  auto* fuse_qkv_transpose_x_shape_node =
      pattern->NewNode(fuse_qkv_transpose_x_shape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto* fuse_qkv_transpose_out_node =
      pattern->NewNode(fuse_qkv_transpose_out_repr())
          ->assert_is_op_output("transpose2", "Out");
  fuse_qkv_reshape_out_node->AsIntermediate()->assert_is_op_input("transpose2",
                                                                  "X");
  fuse_qkv_transpose_node->LinksFrom({fuse_qkv_reshape_out_node})
      .LinksTo({fuse_qkv_transpose_x_shape_node, fuse_qkv_transpose_out_node});

  auto* fuse_qkv_split_node =
      pattern->NewNode(fuse_qkv_split_op_repr())->assert_is_op("split");
  auto* fuse_qkv_split_out_q_node =
      pattern->NewNode(fuse_qkv_split_out_q_repr())
          ->assert_is_op_output("split", "Out");
  auto* fuse_qkv_split_out_k_node =
      pattern->NewNode(fuse_qkv_split_out_k_repr())
          ->assert_is_op_output("split", "Out");
  auto* fuse_qkv_split_out_v_node =
      pattern->NewNode(fuse_qkv_split_out_v_repr())
          ->assert_is_op_output("split", "Out");
  fuse_qkv_transpose_out_node->AsIntermediate()->assert_is_op_input("split",
                                                                    "X");
  fuse_qkv_split_node->LinksFrom({fuse_qkv_transpose_out_node})
      .LinksTo({fuse_qkv_split_out_q_node,
                fuse_qkv_split_out_k_node,
                fuse_qkv_split_out_v_node});

  // core attention pattern
  auto* qk_matmul_node =
      pattern->NewNode(qk_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out_node = pattern->NewNode(qk_matmul_out_repr())
                                 ->assert_is_op_output("matmul_v2", "Out");
  fuse_qkv_split_out_q_node->AsIntermediate()->assert_is_op_input("matmul_v2",
                                                                  "X");
  fuse_qkv_split_out_k_node->AsIntermediate()->assert_is_op_input("matmul_v2",
                                                                  "Y");
  qk_matmul_node
      ->LinksFrom({fuse_qkv_split_out_q_node, fuse_qkv_split_out_k_node})
      .LinksTo({qk_matmul_out_node});

  auto* qk_scale_node =
      pattern->NewNode(qk_scale_op_repr())->assert_is_op("scale");
  auto* qk_scale_out_node = pattern->NewNode(qk_scale_out_repr())
                                ->assert_is_op_output("scale", "Out");
  qk_matmul_out_node->AsIntermediate()->assert_is_op_input("scale", "X");
  qk_scale_node->LinksFrom({qk_matmul_out_node}).LinksTo({qk_scale_out_node});

  PDNode* add_mask_ele_add_out_node{nullptr};
  if (has_attn_mask) {
    auto* add_mask_ele_add_node = pattern->NewNode(add_mask_ele_add_op_repr())
                                      ->assert_is_op("elementwise_add");
    auto* add_mask_ele_add_mask_node =
        pattern->NewNode(add_mask_ele_add_mask_repr())
            ->assert_is_op_input("elementwise_add", "Y");
    add_mask_ele_add_out_node =
        pattern->NewNode(add_mask_ele_add_out_repr())
            ->assert_is_op_output("elementwise_add", "Out");
    qk_scale_out_node->AsIntermediate()->assert_is_op_input("elementwise_add",
                                                            "X");
    add_mask_ele_add_node
        ->LinksFrom({qk_scale_out_node, add_mask_ele_add_mask_node})
        .LinksTo({add_mask_ele_add_out_node});
  }

  auto* qk_softmax_node =
      pattern->NewNode(qk_softmax_op_repr())->assert_is_op("softmax");
  auto* qk_softmax_out_node = pattern->NewNode(qk_softmax_out_repr())
                                  ->assert_is_op_output("softmax", "Out");
  if (has_attn_mask) {
    add_mask_ele_add_out_node->AsIntermediate()->assert_is_op_input("softmax",
                                                                    "X");
    qk_softmax_node->LinksFrom({add_mask_ele_add_out_node})
        .LinksTo({qk_softmax_out_node});
  } else {
    qk_scale_out_node->AsIntermediate()->assert_is_op_input("softmax", "X");
    qk_softmax_node->LinksFrom({qk_scale_out_node})
        .LinksTo({qk_softmax_out_node});
  }

  PDNode* attn_dropout_out_node{nullptr};
  if (do_dropout) {
    auto* attn_dropout_node =
        pattern->NewNode(attn_dropout_op_repr())->assert_is_op("dropout");
    auto* attn_dropout_mask_node = pattern->NewNode(attn_dropout_mask_repr())
                                       ->assert_is_op_output("dropout", "Mask");
    attn_dropout_out_node = pattern->NewNode(attn_dropout_out_repr())
                                ->assert_is_op_output("dropout", "Out");
    qk_softmax_out_node->AsIntermediate()->assert_is_op_input("dropout", "X");
    attn_dropout_node->LinksFrom({qk_softmax_out_node})
        .LinksTo({attn_dropout_mask_node, attn_dropout_out_node});
  }

  auto* qkv_matmul_node =
      pattern->NewNode(qkv_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qkv_matmul_out_node = pattern->NewNode(qkv_matmul_out_repr())
                                  ->assert_is_op_output("matmul_v2", "Out");
  fuse_qkv_split_out_v_node->AsIntermediate()->assert_is_op_input("matmul_v2",
                                                                  "Y");
  if (do_dropout) {
    attn_dropout_out_node->AsIntermediate()->assert_is_op_input("matmul_v2",
                                                                "X");
    qkv_matmul_node
        ->LinksFrom({attn_dropout_out_node, fuse_qkv_split_out_v_node})
        .LinksTo({qkv_matmul_out_node});
  } else {
    qk_softmax_out_node->AsIntermediate()->assert_is_op_input("matmul_v2", "X");
    qkv_matmul_node->LinksFrom({qk_softmax_out_node, fuse_qkv_split_out_v_node})
        .LinksTo({qkv_matmul_out_node});
  }

  auto* qkv_transpose_node =
      pattern->NewNode(qkv_transpose_op_repr())->assert_is_op("transpose2");
  auto* qkv_transpose_x_shape_node =
      pattern->NewNode(qkv_transpose_x_shape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto* qkv_transpose_out_node = pattern->NewNode(qkv_transpose_out_repr())
                                     ->assert_is_op_output("transpose2", "Out");
  qkv_matmul_out_node->AsIntermediate()->assert_is_op_input("transpose2", "X");
  qkv_transpose_node->LinksFrom({qkv_matmul_out_node})
      .LinksTo({qkv_transpose_x_shape_node, qkv_transpose_out_node});

  auto* qkv_reshape_node =
      pattern->NewNode(qkv_reshape_op_repr())->assert_is_op("reshape2");
  auto* qkv_reshape_x_shape_node =
      pattern->NewNode(qkv_reshape_x_shape_repr())
          ->assert_is_op_output("reshape2", "XShape");
  auto* qkv_reshape_out_node = pattern->NewNode(qkv_reshape_out_repr())
                                   ->assert_is_op_output("reshape2", "Out");
  qkv_transpose_out_node->AsIntermediate()->assert_is_op_input("reshape2", "X");
  qkv_reshape_node->LinksFrom({qkv_transpose_out_node})
      .LinksTo({qkv_reshape_x_shape_node, qkv_reshape_out_node});

  // out linear pattern
  auto* out_linear_matmul_node =
      pattern->NewNode(out_linear_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* out_linear_matmul_w_node = pattern->NewNode(out_linear_matmul_w_repr())
                                       ->assert_is_op_input("matmul_v2", "Y");
  auto* out_linear_matmul_out_node =
      pattern->NewNode(out_linear_matmul_out_repr())
          ->assert_is_op_output("matmul_v2", "Out");
  qkv_reshape_out_node->AsIntermediate()->assert_is_op_input("matmul_v2", "X");
  out_linear_matmul_node
      ->LinksFrom({qkv_reshape_out_node, out_linear_matmul_w_node})
      .LinksTo({out_linear_matmul_out_node});

  auto* out_linear_ele_add_node = pattern->NewNode(out_linear_ele_add_op_repr())
                                      ->assert_is_op("elementwise_add");
  auto* out_linear_ele_add_bias_node =
      pattern->NewNode(out_linear_ele_add_bias_repr())
          ->assert_is_op_input("elementwise_add", "Y");
  auto* out_linear_ele_add_out_node =
      pattern->NewNode(out_linear_ele_add_out_repr())
          ->assert_is_op_output("elementwise_add", "Out");
  out_linear_matmul_out_node->AsIntermediate()->assert_is_op_input(
      "elementwise_add", "X");
  out_linear_ele_add_node
      ->LinksFrom({out_linear_matmul_out_node, out_linear_ele_add_bias_node})
      .LinksTo({out_linear_ele_add_out_node});

  auto* out_linear_dropout_node =
      pattern->NewNode(out_linear_dropout_op_repr())->assert_is_op("dropout");
  auto* out_linear_dropout_mask_node =
      pattern->NewNode(out_linear_dropout_mask_repr())
          ->assert_is_op_output("dropout", "Mask");
  auto* out_linear_dropout_out_node =
      pattern->NewNode(out_linear_dropout_out_repr())
          ->assert_is_op_output("dropout", "Out");
  out_linear_ele_add_out_node->AsIntermediate()->assert_is_op_input("dropout",
                                                                    "X");
  out_linear_dropout_node->LinksFrom({out_linear_ele_add_out_node})
      .LinksTo({out_linear_dropout_mask_node, out_linear_dropout_out_node});

  if (!add_residual && !post_layer_norm) {
    return out_linear_dropout_out_node;
  }

  // add residual
  PDNode* residual_ele_add_out_node{nullptr};
  if (add_residual) {
    // this kind of pattern only support `residual + dropout_out`, since we have
    // to fix X and Y
    auto* residual_ele_add_node = pattern->NewNode(residual_ele_add_op_repr())
                                      ->assert_is_op("elementwise_add");
    out_linear_dropout_out_node->AsIntermediate()->assert_is_op_input(
        "elementwise_add", "Y");
    residual_ele_add_node->LinksFrom({x, out_linear_dropout_out_node})
        .LinksTo({residual_ele_add_out_node});

    if (!post_layer_norm) {
      return residual_ele_add_out_node;
    }
  }

  // post layer norm
  auto* post_layer_norm_node =
      pattern->NewNode(post_layer_norm_op_repr())->assert_is_op("layer_norm");
  auto* post_layer_norm_scale_node =
      pattern->NewNode(post_layer_norm_scale_repr())
          ->assert_is_op_input("layer_norm", "Scale");
  auto* post_layer_norm_bias_node =
      pattern->NewNode(post_layer_norm_bias_repr())
          ->assert_is_op_input("layer_norm", "Bias");
  auto* post_layer_norm_out_node = pattern->NewNode(post_layer_norm_out_repr())
                                       ->assert_is_op_output("layer_norm", "Y");
  auto* post_layer_norm_mean_node =
      pattern->NewNode(post_layer_norm_mean_repr())
          ->assert_is_op_output("layer_norm", "Mean");
  auto* post_layer_norm_variance_node =
      pattern->NewNode(post_layer_norm_variance_repr())
          ->assert_is_op_output("layer_norm", "Variance");
  if (add_residual) {
    residual_ele_add_out_node->AsIntermediate()->assert_is_op_input(
        "layer_norm", "X");
    post_layer_norm_node
        ->LinksFrom({residual_ele_add_out_node,
                     post_layer_norm_scale_node,
                     post_layer_norm_bias_node})
        .LinksTo({post_layer_norm_out_node,
                  post_layer_norm_mean_node,
                  post_layer_norm_variance_node});
  } else {
    out_linear_dropout_out_node->AsIntermediate()->assert_is_op_input(
        "layer_norm", "X");
    post_layer_norm_node
        ->LinksFrom({out_linear_dropout_out_node,
                     post_layer_norm_scale_node,
                     post_layer_norm_bias_node})
        .LinksTo({post_layer_norm_out_node,
                  post_layer_norm_mean_node,
                  post_layer_norm_variance_node});
  }

  return post_layer_norm_out_node;
}

PDNode* FusedAttentionGradPattern::operator()(PDNode* x,
                                              bool pre_layer_norm,
                                              bool post_layer_norm,
                                              bool has_attn_mask,
                                              bool do_dropout,
                                              bool add_residual) {
  // TODO(Yuang Liu): finish the backward pattern
  return nullptr;
}

}  // namespace patterns

void FusedAttentionsPass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  graph = PreMaskDropResFwd(graph);
  graph = PreMaskDropResBwd(graph);
}

ir::Graph* FusedAttentionsPass::PreMaskDropResFwd(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(name_scope_, "x"))
                ->AsInput()
                ->assert_is_op_input("layer_norm", "X");
  patterns::FusedAttentionPattern fused_attention_pattern(
      gpd.mutable_pattern(), "fused_attention_pattern");

  fused_attention_pattern(x,
                          /* pre_layer_norm */ true,
                          /* post_layer_norm */ false,
                          /* has_attn_mask */ true,
                          /* do_dropout */ true,
                          /* add_residual */ true);

  int found_fused_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // TODO(Yuang Liu): finish the handler
    VLOG(3) << "handle FusedMultiHeadAttention pass's fusion";
    found_fused_attention++;
  };

  gpd(graph, handler);
  AddStatis(found_fused_attention);

  return graph;
}

ir::Graph* FusedAttentionsPass::PreMaskDropResBwd(Graph* graph) const {
  // TODO(Yuang Liu): finish the pass
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_attention_pass, paddle::framework::ir::FusedAttentionsPass);
