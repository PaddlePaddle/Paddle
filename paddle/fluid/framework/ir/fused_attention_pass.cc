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
  auto* fuse_qkv_matmul_out_node = pattern->NewNode(fuse_qkv_matmul_out_repr())
                                       ->assert_is_op_output("matmul_v2");
  if (pre_layer_norm) {
    pre_layer_norm_out_node->assert_is_op_input("matmul_v2", "X");
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
          ->assert_is_op_output("elementwise_add");
  fuse_qkv_matmul_out_node->assert_is_op_input("elementwise_add", "X");
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
          ->assert_is_op_output("reshape2");
  fuse_qkv_ele_add_out_node->assert_is_op_input("reshape2", "X");
  fuse_qkv_reshape_node->LinksFrom({fuse_qkv_ele_add_out_node})
      .LinksTo({fuse_qkv_reshape_x_shape_node, fuse_qkv_reshape_out_node});

  auto* fuse_qkv_transpose_node = pattern->NewNode(fuse_qkv_transpose_op_repr())
                                      ->assert_is_op("transpose2");
  auto* fuse_qkv_transpose_x_shape_node =
      pattern->NewNode(fuse_qkv_transpose_x_shape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto* fuse_qkv_transpose_out_node =
      pattern->NewNode(fuse_qkv_transpose_out_repr())
          ->assert_is_op_output("transpose2");
  fuse_qkv_reshape_out_node->assert_is_op_input("transpose2", "X");
  fuse_qkv_transpose_node->LinksFrom({fuse_qkv_reshape_out_node})
      .LinksTo({fuse_qkv_transpose_x_shape_node, fuse_qkv_transpose_out_node});

  auto* fuse_qkv_split_node =
      pattern->NewNode(fuse_qkv_split_op_repr())->assert_is_op("split");
  auto* fuse_qkv_split_out_q_node =
      pattern->NewNode(fuse_qkv_split_out_q_repr())
          ->assert_is_op_output("split");
  auto* fuse_qkv_split_out_k_node =
      pattern->NewNode(fuse_qkv_split_out_k_repr())
          ->assert_is_op_output("split");
  auto* fuse_qkv_split_out_v_node =
      pattern->NewNode(fuse_qkv_split_out_v_repr())
          ->assert_is_op_output("split");
  fuse_qkv_transpose_out_node->assert_is_op_input("split", "X");
  fuse_qkv_split_node->LinksFrom({fuse_qkv_transpose_out_node})
      .LinksTo({fuse_qkv_split_out_q_node,
                fuse_qkv_split_out_k_node,
                fuse_qkv_split_out_v_node});

  // core attention pattern
  auto* qk_matmul_node =
      pattern->NewNode(qk_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qk_matmul_out_node =
      pattern->NewNode(qk_matmul_out_repr())->assert_is_op_output("matmul_v2");
  fuse_qkv_split_out_q_node->assert_is_op_input("matmul_v2", "X");
  fuse_qkv_split_out_k_node->assert_is_op_input("matmul_v2", "Y");
  qk_matmul_node
      ->LinksFrom({fuse_qkv_split_out_q_node, fuse_qkv_split_out_k_node})
      .LinksTo({qk_matmul_out_node});

  auto* qk_scale_node =
      pattern->NewNode(qk_scale_op_repr())->assert_is_op("scale");
  auto* qk_scale_out_node =
      pattern->NewNode(qk_scale_out_repr())->assert_is_op_output("scale");
  qk_matmul_out_node->assert_is_op_input("scale", "X");
  qk_scale_node->LinksFrom({qk_matmul_out_node}).LinksTo({qk_scale_out_node});

  PDNode* add_mask_ele_add_out_node{nullptr};
  if (has_attn_mask) {
    auto* add_mask_ele_add_node = pattern->NewNode(add_mask_ele_add_op_repr())
                                      ->assert_is_op("elementwise_add");
    auto* add_mask_ele_add_mask_node =
        pattern->NewNode(add_mask_ele_add_mask_repr())
            ->assert_is_op_input("elementwise_add", "Y");
    add_mask_ele_add_out_node = pattern->NewNode(add_mask_ele_add_out_repr())
                                    ->assert_is_op_output("elementwise_add");
    qk_scale_out_node->assert_is_op_input("elementwise_add", "X");
    add_mask_ele_add_node
        ->LinksFrom({qk_scale_out_node, add_mask_ele_add_mask_node})
        .LinksTo({add_mask_ele_add_out_node});
  }

  auto* qk_softmax_node =
      pattern->NewNode(qk_softmax_op_repr())->assert_is_op("softmax");
  auto* qk_softmax_out_node =
      pattern->NewNode(qk_softmax_out_repr())->assert_is_op_output("softmax");
  if (has_attn_mask) {
    add_mask_ele_add_out_node->assert_is_op_input("softmax", "X");
    qk_softmax_node->LinksFrom({add_mask_ele_add_out_node})
        .LinksTo({qk_softmax_out_node});
  } else {
    qk_scale_out_node->assert_is_op_input("softmax", "X");
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
                                ->assert_is_op_output("dropout");
    qk_softmax_out_node->assert_is_op_input("dropout", "X");
    attn_dropout_node->LinksFrom({qk_softmax_out_node})
        .LinksTo({attn_dropout_mask_node, attn_dropout_out_node});
  }

  auto* qkv_matmul_node =
      pattern->NewNode(qkv_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* qkv_matmul_out_node =
      pattern->NewNode(qkv_matmul_out_repr())->assert_is_op_output("matmul_v2");
  fuse_qkv_split_out_v_node->assert_is_op_input("matmul_v2", "Y");
  if (do_dropout) {
    attn_dropout_out_node->assert_is_op_input("matmul_v2", "X");
    qkv_matmul_node
        ->LinksFrom({attn_dropout_out_node, fuse_qkv_split_out_v_node})
        .LinksTo({qkv_matmul_out_node});
  } else {
    qk_softmax_out_node->assert_is_op_input("matmul_v2", "X");
    qkv_matmul_node->LinksFrom({qk_softmax_out_node, fuse_qkv_split_out_v_node})
        .LinksTo({qkv_matmul_out_node});
  }

  auto* qkv_transpose_node =
      pattern->NewNode(qkv_transpose_op_repr())->assert_is_op("transpose2");
  auto* qkv_transpose_x_shape_node =
      pattern->NewNode(qkv_transpose_x_shape_repr())
          ->assert_is_op_output("transpose2", "XShape");
  auto* qkv_transpose_out_node = pattern->NewNode(qkv_transpose_out_repr())
                                     ->assert_is_op_output("transpose2");
  qkv_matmul_out_node->assert_is_op_input("transpose2", "X");
  qkv_transpose_node->LinksFrom({qkv_matmul_out_node})
      .LinksTo({qkv_transpose_x_shape_node, qkv_transpose_out_node});

  auto* qkv_reshape_node =
      pattern->NewNode(qkv_reshape_op_repr())->assert_is_op("reshape2");
  auto* qkv_reshape_x_shape_node =
      pattern->NewNode(qkv_reshape_x_shape_repr())
          ->assert_is_op_output("reshape2", "XShape");
  auto* qkv_reshape_out_node =
      pattern->NewNode(qkv_reshape_out_repr())->assert_is_op_output("reshape2");
  qkv_transpose_out_node->assert_is_op_input("reshape2", "X");
  qkv_reshape_node->LinksFrom({qkv_transpose_out_node})
      .LinksTo({qkv_reshape_x_shape_node, qkv_reshape_out_node});

  // out linear pattern
  auto* out_linear_matmul_node =
      pattern->NewNode(out_linear_matmul_op_repr())->assert_is_op("matmul_v2");
  auto* out_linear_matmul_w_node = pattern->NewNode(out_linear_matmul_w_repr())
                                       ->assert_is_op_input("matmul_v2", "Y");
  auto* out_linear_matmul_out_node =
      pattern->NewNode(out_linear_matmul_out_repr())
          ->assert_is_op_output("matmul_v2");
  qkv_reshape_out_node->assert_is_op_input("matmul_v2", "X");
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
          ->assert_is_op_output("elementwise_add");
  out_linear_matmul_out_node->assert_is_op_input("elementwise_add", "X");
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
          ->assert_is_op_output("dropout");
  out_linear_ele_add_out_node->assert_is_op_input("dropout", "X");
  out_linear_dropout_node->LinksFrom({out_linear_ele_add_out_node})
      .LinksTo({out_linear_dropout_mask_node, out_linear_dropout_out_node});

  if (!add_residual && pre_layer_norm) {
    return out_linear_dropout_out_node;
  }

  // add residual
  PDNode* residual_ele_add_out_node{nullptr};
  if (add_residual) {
    // this kind of pattern only support `residual + dropout_out`, since we have
    // to fix X and Y
    auto* residual_ele_add_node = pattern->NewNode(residual_ele_add_op_repr())
                                      ->assert_is_op("elementwise_add");
    residual_ele_add_out_node = pattern->NewNode(residual_ele_add_out_repr())
                                    ->assert_is_op_output("elementwise_add");
    out_linear_dropout_out_node->assert_is_op_input("elementwise_add", "Y");
    residual_ele_add_node->LinksFrom({x, out_linear_dropout_out_node})
        .LinksTo({residual_ele_add_out_node});

    if (pre_layer_norm) {
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
    residual_ele_add_out_node->assert_is_op_input("layer_norm", "X");
    post_layer_norm_node
        ->LinksFrom({residual_ele_add_out_node,
                     post_layer_norm_scale_node,
                     post_layer_norm_bias_node})
        .LinksTo({post_layer_norm_out_node,
                  post_layer_norm_mean_node,
                  post_layer_norm_variance_node});
  } else {
    out_linear_dropout_out_node->assert_is_op_input("layer_norm", "X");
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
                                              bool has_attn_mask,
                                              bool do_dropout,
                                              bool add_residual) {
  // post layer norm
  PDNode* post_layer_norm_grad_out_node{nullptr};
  if (!pre_layer_norm) {
    auto* post_layer_norm_grad_node =
        pattern->NewNode(post_layer_norm_grad_op_repr())
            ->assert_is_op("layer_norm_grad");
    auto* post_layer_norm_grad_bias_node =
        pattern->NewNode(post_layer_norm_grad_bias_repr())
            ->assert_is_op_input("layer_norm_grad", "Bias");
    auto* post_layer_norm_grad_scale_node =
        pattern->NewNode(post_layer_norm_grad_scale_repr())
            ->assert_is_op_input("layer_norm_grad", "Scale");
    auto* post_layer_norm_grad_mean_node =
        pattern->NewNode(post_layer_norm_grad_mean_repr())
            ->assert_is_op_input("layer_norm_grad", "Mean");
    auto* post_layer_norm_grad_variance_node =
        pattern->NewNode(post_layer_norm_grad_variance_repr())
            ->assert_is_op_input("layer_norm_grad", "Variance");
    auto* post_layer_norm_grad_x_node =
        pattern->NewNode(post_layer_norm_grad_x_repr())
            ->assert_is_op_input("layer_norm_grad", "X");
    post_layer_norm_grad_out_node =
        pattern->NewNode(post_layer_norm_grad_x_grad_repr())
            ->assert_is_op_output("layer_norm_grad", "X@GRAD");
    auto* post_layer_norm_grad_bias_grad_node =
        pattern->NewNode(post_layer_norm_grad_bias_grad_repr())
            ->assert_is_op_output("layer_norm_grad", "Bias@GRAD");
    auto* post_layer_norm_grad_scale_grad_node =
        pattern->NewNode(post_layer_norm_grad_scale_grad_repr())
            ->assert_is_op_output("layer_norm_grad", "Scale@GRAD");
    post_layer_norm_grad_node
        ->LinksFrom({x,
                     post_layer_norm_grad_bias_node,
                     post_layer_norm_grad_scale_node,
                     post_layer_norm_grad_mean_node,
                     post_layer_norm_grad_variance_node,
                     post_layer_norm_grad_x_node})
        .LinksTo({post_layer_norm_grad_out_node,
                  post_layer_norm_grad_bias_grad_node,
                  post_layer_norm_grad_scale_grad_node});
  }

  // add residual
  PDNode* residual_ele_add_grad_out_node{nullptr};
  PDNode* residual_ele_add_grad_x_node{nullptr};
  PDNode* residual_ele_add_grad_x_grad_node{nullptr};
  if (add_residual) {
    PDNode* ele_add_grad_input = x;
    if (!pre_layer_norm) {
      ele_add_grad_input = post_layer_norm_grad_out_node;
    }
    auto* residual_ele_add_grad_node =
        pattern->NewNode(residual_ele_add_grad_op_repr())
            ->assert_is_op("elementwise_add_grad");
    residual_ele_add_grad_x_node =
        pattern->NewNode(residual_ele_add_grad_x_repr())
            ->assert_is_op_input("elementwise_add_grad", "X");
    auto* residual_ele_add_grad_bias_node =
        pattern->NewNode(residual_ele_add_grad_bias_repr())
            ->assert_is_op_input("elementwise_add_grad", "Y");
    residual_ele_add_grad_out_node =
        pattern->NewNode(residual_ele_add_grad_bias_grad_repr())
            ->assert_is_op_output("elementwise_add_grad", "Y@GRAD");
    residual_ele_add_grad_x_grad_node =
        pattern->NewNode(residual_ele_add_grad_x_grad_repr())
            ->assert_is_op_output("elementwise_add_grad", "X@GRAD");
    ele_add_grad_input->assert_is_op_input("elementwise_add_grad", "Out@GRAD");
    residual_ele_add_grad_node
        ->LinksFrom({ele_add_grad_input,
                     residual_ele_add_grad_x_node,
                     residual_ele_add_grad_bias_node})
        .LinksTo({residual_ele_add_grad_x_grad_node,
                  residual_ele_add_grad_out_node});
  }

  // get the real input x for dropout grad
  PDNode* out_linear_grad_input_node = x;
  if (!pre_layer_norm && !add_residual) {
    out_linear_grad_input_node = post_layer_norm_grad_out_node;
  } else if (add_residual) {
    out_linear_grad_input_node = residual_ele_add_grad_out_node;
  }

  // out linear part
  auto* out_linear_dropout_grad_node =
      pattern->NewNode(out_linear_dropout_grad_op_repr())
          ->assert_is_op("dropout_grad");
  auto* out_linear_dropout_grad_mask_node =
      pattern->NewNode(out_linear_dropout_grad_mask_repr())
          ->assert_is_op_input("dropout_grad", "Mask");
  auto* out_linear_dropout_grad_out_node =
      pattern->NewNode(out_linear_dropout_grad_out_repr())
          ->assert_is_op_output("dropout_grad", "X@GRAD");
  out_linear_grad_input_node->assert_is_op_input("dropout_grad", "Out@GRAD");
  out_linear_dropout_grad_node
      ->LinksFrom(
          {out_linear_grad_input_node, out_linear_dropout_grad_mask_node})
      .LinksTo({out_linear_dropout_grad_out_node});

  auto* out_linear_ele_add_grad_node =
      pattern->NewNode(out_linear_ele_add_grad_op_repr())
          ->assert_is_op("elementwise_add_grad");
  auto* out_linear_ele_add_grad_x_node =
      pattern->NewNode(out_linear_ele_add_grad_x_repr())
          ->assert_is_op_input("elementwise_add_grad", "X");
  auto* out_linear_ele_add_grad_bias_node =
      pattern->NewNode(out_linear_ele_add_grad_bias_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y");
  auto* out_linear_ele_add_grad_x_grad_node =
      pattern->NewNode(out_linear_ele_add_grad_x_grad_repr())
          ->assert_is_op_output("elementwise_add_grad", "X@GRAD");
  auto* out_linear_ele_add_grad_bias_grad_node =
      pattern->NewNode(out_linear_ele_add_grad_bias_grad_repr())
          ->assert_is_op_output("elementwise_add_grad", "Y@GRAD");
  out_linear_dropout_grad_out_node->assert_is_op_input("elementwise_add_grad",
                                                       "Out@GRAD");
  out_linear_ele_add_grad_node
      ->LinksFrom({out_linear_dropout_grad_out_node,
                   out_linear_ele_add_grad_x_node,
                   out_linear_ele_add_grad_bias_node})
      .LinksTo({out_linear_ele_add_grad_x_grad_node,
                out_linear_ele_add_grad_bias_grad_node});

  auto* out_linear_matmul_grad_node =
      pattern->NewNode(out_linear_matmul_grad_op_repr())
          ->assert_is_op("matmul_v2_grad");
  auto* out_linear_matmul_grad_x_node =
      pattern->NewNode(out_linear_matmul_grad_x_repr())
          ->assert_is_op_input("matmul_v2_grad", "X");
  auto* out_linear_matmul_grad_w_node =
      pattern->NewNode(out_linear_matmul_grad_w_repr())
          ->assert_is_op_input("matmul_v2_grad", "Y");
  auto* out_linear_matmul_grad_x_grad_node =
      pattern->NewNode(out_linear_matmul_grad_x_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "X@GRAD");
  auto* out_linear_matmul_grad_w_grad_node =
      pattern->NewNode(out_linear_matmul_grad_w_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "Y@GRAD");
  out_linear_ele_add_grad_x_grad_node->assert_is_op_input("matmul_v2_grad",
                                                          "Out@GRAD");
  out_linear_matmul_grad_node
      ->LinksFrom({out_linear_ele_add_grad_x_grad_node,
                   out_linear_matmul_grad_x_node,
                   out_linear_matmul_grad_w_node})
      .LinksTo({out_linear_matmul_grad_x_grad_node,
                out_linear_matmul_grad_w_grad_node});

  // core attention part
  auto* qkv_reshape_grad_node = pattern->NewNode(qkv_reshape_grad_op_repr())
                                    ->assert_is_op("reshape2_grad");
  auto* qkv_reshape_grad_x_shape_node =
      pattern->NewNode(qkv_reshape_grad_x_shape_repr())
          ->assert_is_op_input("reshape2_grad", "XShape");
  auto* qkv_reshape_grad_out_node =
      pattern->NewNode(qkv_reshape_grad_out_repr())
          ->assert_is_op_output("reshape2_grad", "X@GRAD");
  out_linear_matmul_grad_x_grad_node->assert_is_op_input("reshape2_grad",
                                                         "Out@GRAD");
  qkv_reshape_grad_node
      ->LinksFrom(
          {out_linear_matmul_grad_x_grad_node, qkv_reshape_grad_x_shape_node})
      .LinksTo({qkv_reshape_grad_out_node});

  auto* qkv_transpose_grad_node = pattern->NewNode(qkv_transpose_grad_op_repr())
                                      ->assert_is_op("transpose2_grad");
  auto* qkv_transpose_grad_x_shape_node =
      pattern->NewNode(qkv_transpose_grad_x_shape_repr())
          ->assert_is_op_input("transpose2_grad", "XShape");
  auto* qkv_transpose_grad_out_node =
      pattern->NewNode(qkv_transpose_grad_out_repr())
          ->assert_is_op_output("transpose2_grad", "X@GRAD");
  qkv_reshape_grad_out_node->assert_is_op_input("transpose2_grad", "Out@GRAD");
  qkv_transpose_grad_node
      ->LinksFrom({qkv_reshape_grad_out_node, qkv_transpose_grad_x_shape_node})
      .LinksTo({qkv_transpose_grad_out_node});

  auto* qkv_matmul_grad_node = pattern->NewNode(qkv_matmul_grad_op_repr())
                                   ->assert_is_op("matmul_v2_grad");
  auto* qkv_matmul_grad_x_node =
      pattern->NewNode(qkv_matmul_grad_x_repr())
          ->assert_is_op_input("matmul_v2_grad", "X");
  auto* qkv_matmul_grad_w_node =
      pattern->NewNode(qkv_matmul_grad_w_repr())
          ->assert_is_op_input("matmul_v2_grad", "Y");
  auto* qkv_matmul_grad_x_grad_node =
      pattern->NewNode(qkv_matmul_grad_x_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "X@GRAD");
  auto* qkv_matmul_grad_w_grad_node =
      pattern->NewNode(qkv_matmul_grad_w_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "Y@GRAD");
  qkv_transpose_grad_out_node->assert_is_op_input("matmul_v2_grad", "Out@GRAD");
  qkv_matmul_grad_node
      ->LinksFrom({qkv_transpose_grad_out_node,
                   qkv_matmul_grad_x_node,
                   qkv_matmul_grad_w_node})
      .LinksTo({qkv_matmul_grad_x_grad_node, qkv_matmul_grad_w_grad_node});

  PDNode* attn_dropout_grad_out_node{nullptr};
  if (do_dropout) {
    auto* attn_dropout_grad_node = pattern->NewNode(attn_dropout_grad_op_repr())
                                       ->assert_is_op("dropout_grad");
    auto* attn_dropout_grad_mask_node =
        pattern->NewNode(attn_dropout_grad_mask_repr())
            ->assert_is_op_input("dropout_grad", "Mask");
    attn_dropout_grad_out_node =
        pattern->NewNode(attn_dropout_grad_out_repr())
            ->assert_is_op_output("dropout_grad", "X@GRAD");
    qkv_matmul_grad_x_grad_node->assert_is_op_input("dropout_grad", "Out@GRAD");
    attn_dropout_grad_node
        ->LinksFrom({qkv_matmul_grad_x_grad_node, attn_dropout_grad_mask_node})
        .LinksTo({attn_dropout_grad_out_node});
  }

  PDNode* qk_softmax_grad_input_node =
      do_dropout ? attn_dropout_grad_out_node : qkv_matmul_grad_x_grad_node;
  auto* qk_softmax_grad_node =
      pattern->NewNode(qk_softmax_grad_op_repr())->assert_is_op("softmax_grad");
  auto* qk_softmax_grad_fwd_out_node =
      pattern->NewNode(qk_softmax_grad_fwd_out_repr())
          ->assert_is_op_input("softmax_grad", "Out");
  auto* qk_softmax_grad_out =
      pattern->NewNode(qk_softmax_grad_out_repr())
          ->assert_is_op_output("softmax_grad", "X@GRAD");
  qk_softmax_grad_input_node->assert_is_op_input("softmax_grad", "Out@GRAD");
  qk_softmax_grad_node
      ->LinksFrom({qk_softmax_grad_input_node, qk_softmax_grad_fwd_out_node})
      .LinksTo({qk_softmax_grad_out});

  PDNode* add_mask_ele_add_grad_x_grad_node{nullptr};
  if (has_attn_mask) {
    auto* add_mask_ele_add_grad_node =
        pattern->NewNode(add_mask_ele_add_grad_op_repr())
            ->assert_is_op("elementwise_add_grad");
    auto* add_mask_ele_add_grad_x_node =
        pattern->NewNode(add_mask_ele_add_grad_x_repr())
            ->assert_is_op_input("elementwise_add_grad", "X");
    auto* add_mask_ele_add_grad_bias_node =
        pattern->NewNode(add_mask_ele_add_grad_bias_repr())
            ->assert_is_op_input("elementwise_add_grad", "Y");
    add_mask_ele_add_grad_x_grad_node =
        pattern->NewNode(add_mask_ele_add_grad_x_grad_repr())
            ->assert_is_op_output("elementwise_add_grad", "X@GRAD");
    qk_softmax_grad_out->assert_is_op_input("elementwise_add_grad", "Out@GRAD");
    add_mask_ele_add_grad_node
        ->LinksFrom({add_mask_ele_add_grad_x_node,
                     add_mask_ele_add_grad_bias_node,
                     qk_softmax_grad_out})
        .LinksTo({add_mask_ele_add_grad_x_grad_node});
  }

  PDNode* qk_scale_grad_input_node =
      has_attn_mask ? add_mask_ele_add_grad_x_grad_node : qk_softmax_grad_out;
  auto* qk_scale_grad_node =
      pattern->NewNode(qk_scale_grad_op_repr())->assert_is_op("scale");
  auto* qk_scale_grad_out_node =
      pattern->NewNode(qk_scale_grad_out_repr())->assert_is_op_output("scale");
  qk_scale_grad_input_node->assert_is_op_input("scale", "X");
  qk_scale_grad_node->LinksFrom({qk_scale_grad_input_node})
      .LinksTo({qk_scale_grad_out_node});

  auto* qk_matmul_grad_node = pattern->NewNode(qk_matmul_grad_op_repr())
                                  ->assert_is_op("matmul_v2_grad");
  auto* qk_matmul_grad_x_node = pattern->NewNode(qk_matmul_grad_x_repr())
                                    ->assert_is_op_input("matmul_v2_grad", "X");
  auto* qk_matmul_grad_w_node = pattern->NewNode(qk_matmul_grad_w_repr())
                                    ->assert_is_op_input("matmul_v2_grad", "Y");
  auto* qk_matmul_grad_x_grad_node =
      pattern->NewNode(qk_matmul_grad_x_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "X@GRAD");
  auto* qk_matmul_grad_w_grad_node =
      pattern->NewNode(qk_matmul_grad_w_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "Y@GRAD");
  qk_scale_grad_out_node->assert_is_op_input("matmul_v2_grad", "Out@GRAD");
  qk_matmul_grad_node
      ->LinksFrom({qk_scale_grad_out_node,
                   qk_matmul_grad_x_node,
                   qk_matmul_grad_w_node})
      .LinksTo({qk_matmul_grad_x_grad_node, qk_matmul_grad_w_grad_node});

  // fuse qkv projection
  auto* fuse_qkv_split_grad_node =
      pattern->NewNode(fuse_qkv_split_grad_op_repr())->assert_is_op("concat");
  auto* fuse_qkv_split_grad_out_node =
      pattern->NewNode(fuse_qkv_split_grad_out_repr())
          ->assert_is_op_output("concat");
  qk_matmul_grad_x_grad_node->assert_is_op_input("concat");   // q grad
  qk_matmul_grad_w_grad_node->assert_is_op_input("concat");   // k grad
  qkv_matmul_grad_w_grad_node->assert_is_op_input("concat");  // v grad
  fuse_qkv_split_grad_node
      ->LinksFrom({qk_matmul_grad_x_grad_node,
                   qk_matmul_grad_w_grad_node,
                   qkv_matmul_grad_w_grad_node})
      .LinksTo({fuse_qkv_split_grad_out_node});

  auto* fuse_qkv_transpose_grad_node =
      pattern->NewNode(fuse_qkv_transpose_grad_op_repr())
          ->assert_is_op("transpose2_grad");
  auto* fuse_qkv_transpose_grad_x_shape_node =
      pattern->NewNode(fuse_qkv_transpose_grad_x_shape_repr())
          ->assert_is_op_input("transpose2_grad", "XShape");
  auto* fuse_qkv_transpose_grad_out_node =
      pattern->NewNode(fuse_qkv_transpose_grad_out_repr())
          ->assert_is_op_output("transpose2_grad", "X@GRAD");
  fuse_qkv_split_grad_out_node->assert_is_op_input("transpose2_grad",
                                                   "Out@GRAD");
  fuse_qkv_transpose_grad_node
      ->LinksFrom(
          {fuse_qkv_split_grad_out_node, fuse_qkv_transpose_grad_x_shape_node})
      .LinksTo({fuse_qkv_transpose_grad_out_node});

  auto* fuse_qkv_reshape_grad_node =
      pattern->NewNode(fuse_qkv_reshape_grad_op_repr())
          ->assert_is_op("reshape2_grad");
  auto* fuse_qkv_reshape_grad_x_shape_node =
      pattern->NewNode(fuse_qkv_reshape_grad_x_shape_repr())
          ->assert_is_op_input("reshape2_grad", "XShape");
  auto* fuse_qkv_reshape_grad_out_node =
      pattern->NewNode(fuse_qkv_reshape_grad_out_repr())
          ->assert_is_op_output("reshape2_grad", "X@GRAD");
  fuse_qkv_transpose_grad_out_node->assert_is_op_input("reshape2_grad",
                                                       "Out@GRAD");
  fuse_qkv_reshape_grad_node
      ->LinksFrom({fuse_qkv_transpose_grad_out_node,
                   fuse_qkv_reshape_grad_x_shape_node})
      .LinksTo({fuse_qkv_reshape_grad_out_node});

  auto* fuse_qkv_ele_add_grad_node =
      pattern->NewNode(fuse_qkv_ele_add_grad_op_repr())
          ->assert_is_op("elementwise_add_grad");
  auto* fuse_qkv_ele_add_grad_x_node =
      pattern->NewNode(fuse_qkv_ele_add_grad_x_repr())
          ->assert_is_op_input("elementwise_add_grad", "X");
  auto* fuse_qkv_ele_add_grad_bias_node =
      pattern->NewNode(fuse_qkv_ele_add_grad_bias_repr())
          ->assert_is_op_input("elementwise_add_grad", "Y");
  auto* fuse_qkv_ele_add_grad_x_grad_node =
      pattern->NewNode(fuse_qkv_ele_add_grad_x_grad_repr())
          ->assert_is_op_output("elementwise_add_grad", "X@GRAD");
  auto* fuse_qkv_ele_add_grad_bias_grad_node =
      pattern->NewNode(fuse_qkv_ele_add_grad_bias_grad_repr())
          ->assert_is_op_output("elementwise_add_grad", "Y@GRAD");
  fuse_qkv_reshape_grad_out_node->assert_is_op_input("elementwise_add_grad",
                                                     "Out@GRAD");
  fuse_qkv_ele_add_grad_node
      ->LinksFrom({fuse_qkv_reshape_grad_out_node,
                   fuse_qkv_ele_add_grad_x_node,
                   fuse_qkv_ele_add_grad_bias_node})
      .LinksTo({fuse_qkv_ele_add_grad_x_grad_node,
                fuse_qkv_ele_add_grad_bias_grad_node});

  auto* fuse_qkv_matmul_grad_node =
      pattern->NewNode(fuse_qkv_matmul_grad_op_repr())
          ->assert_is_op("matmul_v2_grad");
  auto* fuse_qkv_matmul_grad_x_node =
      pattern->NewNode(fuse_qkv_matmul_grad_x_repr())
          ->assert_is_op_input("matmul_v2_grad", "X");
  auto* fuse_qkv_matmul_grad_w_node =
      pattern->NewNode(fuse_qkv_matmul_grad_w_repr())
          ->assert_is_op_input("matmul_v2_grad", "Y");
  auto* fuse_qkv_matmul_grad_x_grad_node =
      pattern->NewNode(fuse_qkv_matmul_grad_x_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "X@GRAD");
  auto* fuse_qkv_matmul_grad_w_grad_node =
      pattern->NewNode(fuse_qkv_matmul_grad_w_grad_repr())
          ->assert_is_op_output("matmul_v2_grad", "Y@GRAD");
  fuse_qkv_ele_add_grad_x_grad_node->assert_is_op_input("matmul_v2_grad",
                                                        "Out@GRAD");
  fuse_qkv_matmul_grad_node
      ->LinksFrom({fuse_qkv_ele_add_grad_x_grad_node,
                   fuse_qkv_matmul_grad_x_node,
                   fuse_qkv_matmul_grad_w_node})
      .LinksTo(
          {fuse_qkv_matmul_grad_x_grad_node, fuse_qkv_matmul_grad_w_grad_node});

  if (!pre_layer_norm) {
    return fuse_qkv_matmul_grad_x_grad_node;
  }

  // pre layer norm
  auto* pre_layer_norm_grad_node =
      pattern->NewNode(pre_layer_norm_grad_op_repr())
          ->assert_is_op("layer_norm_grad");
  auto* pre_layer_norm_grad_scale_node =
      pattern->NewNode(pre_layer_norm_grad_scale_repr())
          ->assert_is_op_input("layer_norm_grad", "Scale");
  auto* pre_layer_norm_grad_bias_node =
      pattern->NewNode(pre_layer_norm_grad_bias_repr())
          ->assert_is_op_input("layer_norm_grad", "Bias");
  auto* pre_layer_norm_grad_mean_node =
      pattern->NewNode(pre_layer_norm_grad_mean_repr())
          ->assert_is_op_input("layer_norm_grad", "Mean");
  auto* pre_layer_norm_grad_variance_node =
      pattern->NewNode(pre_layer_norm_grad_variance_repr())
          ->assert_is_op_input("layer_norm_grad", "Variance");
  auto* pre_layer_norm_grad_x_node =
      add_residual ? residual_ele_add_grad_x_node
                   : pattern->NewNode(pre_layer_norm_grad_x_repr())
                         ->assert_is_op_input("layer_norm_grad", "X");
  auto* pre_layer_norm_grad_scale_grad_node =
      pattern->NewNode(pre_layer_norm_grad_scale_grad_repr())
          ->assert_is_op_output("layer_norm_grad", "Scale@GRAD");
  auto* pre_layer_norm_grad_bias_grad_node =
      pattern->NewNode(pre_layer_norm_grad_bias_grad_repr())
          ->assert_is_op_output("layer_norm_grad", "Bias@GRAD");
  auto* pre_layer_norm_grad_x_grad_node =
      pattern->NewNode(pre_layer_norm_grad_x_grad_repr())
          ->assert_is_op_output("layer_norm_grad", "X@GRAD");
  fuse_qkv_matmul_grad_x_grad_node->assert_is_op_input("layer_norm_grad",
                                                       "Y@GRAD");
  pre_layer_norm_grad_node
      ->LinksFrom({fuse_qkv_matmul_grad_x_grad_node,
                   pre_layer_norm_grad_scale_node,
                   pre_layer_norm_grad_bias_node,
                   pre_layer_norm_grad_mean_node,
                   pre_layer_norm_grad_variance_node,
                   pre_layer_norm_grad_x_node})
      .LinksTo({pre_layer_norm_grad_scale_grad_node,
                pre_layer_norm_grad_bias_grad_node,
                pre_layer_norm_grad_x_grad_node});

  if (!add_residual) {
    return pre_layer_norm_grad_x_grad_node;
  }

  auto* grad_accumulation_sum_node =
      pattern->NewNode(grad_accumulation_sum_op_repr())->assert_is_op("sum");
  auto* grad_accumulation_sum_out_node =
      pattern->NewNode(grad_accumulation_out_repr())
          ->assert_is_op_output("sum");
  grad_accumulation_sum_node
      ->LinksFrom(
          {pre_layer_norm_grad_x_grad_node, residual_ele_add_grad_x_grad_node})
      .LinksTo({grad_accumulation_sum_out_node});

  return grad_accumulation_sum_out_node;
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
                          /* has_attn_mask */ true,
                          /* do_dropout */ true,
                          /* add_residual */ true);

  int found_fused_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "handle FusedMultiHeadAttention pass's fusion";

    GET_IR_NODE_FROM_SUBGRAPH(
        pre_layer_norm_op_node, pre_layer_norm_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fuse_qkv_matmul_op_node, fuse_qkv_matmul_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fuse_qkv_ele_add_op_node, fuse_qkv_ele_add_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fuse_qkv_reshape_op_node, fuse_qkv_reshape_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_transpose_op_node,
                              fuse_qkv_transpose_op,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fuse_qkv_split_op_node, fuse_qkv_split_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_matmul_op_node, qk_matmul_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_scale_op_node, qk_scale_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        add_mask_ele_add_op_node, add_mask_ele_add_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_softmax_op_node, qk_softmax_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dropout_op_node, attn_dropout_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_matmul_op_node, qkv_matmul_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_transpose_op_node, qkv_transpose_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_reshape_op_node, qkv_reshape_op, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_matmul_op_node,
                              out_linear_matmul_op,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_ele_add_op_node,
                              out_linear_ele_add_op,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_dropout_op_node,
                              out_linear_dropout_op,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        residual_ele_add_op_node, residual_ele_add_op, fused_attention_pattern);

    OpDesc fused_attention_op_desc(pre_layer_norm_op_node->Op()->Block());
    fused_attention_op_desc.SetType("fused_attention");
    fused_attention_op_desc.SetInput("X", {subgraph.at(x)->Name()});

    fused_attention_op_desc.SetAttr("pre_layer_norm", true);
    GET_IR_NODE_FROM_SUBGRAPH(pre_layer_norm_scale_node,
                              pre_layer_norm_scale,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        pre_layer_norm_bias_node, pre_layer_norm_bias, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        pre_layer_norm_out_node, pre_layer_norm_out, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        pre_layer_norm_mean_node, pre_layer_norm_mean, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pre_layer_norm_variance_node,
                              pre_layer_norm_variance,
                              fused_attention_pattern);
    fused_attention_op_desc.SetInput("LnScale",
                                     {pre_layer_norm_scale_node->Name()});
    fused_attention_op_desc.SetInput("LnBias",
                                     {pre_layer_norm_bias_node->Name()});
    fused_attention_op_desc.SetOutput("LnOut",
                                      {pre_layer_norm_out_node->Name()});
    fused_attention_op_desc.SetOutput("LnMean",
                                      {pre_layer_norm_mean_node->Name()});
    fused_attention_op_desc.SetOutput("LnVariance",
                                      {pre_layer_norm_variance_node->Name()});
    fused_attention_op_desc.SetAttr(
        "epsilon",
        PADDLE_GET_CONST(float,
                         pre_layer_norm_op_node->Op()->GetAttr("epsilon")));

    fused_attention_op_desc.SetAttr("transpose_qkv_wb", true);
    std::vector<int> shape = PADDLE_GET_CONST(
        std::vector<int>, fuse_qkv_reshape_op_node->Op()->GetAttr("shape"));
    fused_attention_op_desc.SetAttr("num_heads", shape[2]);
    GET_IR_NODE_FROM_SUBGRAPH(
        fuse_qkv_matmul_w_node, fuse_qkv_matmul_w, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        fuse_qkv_matmul_out_node, fuse_qkv_matmul_out, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_ele_add_bias_node,
                              fuse_qkv_ele_add_bias,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_ele_add_out_node,
                              fuse_qkv_ele_add_out,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_transpose_out_node,
                              fuse_qkv_transpose_out,
                              fused_attention_pattern);
    fused_attention_op_desc.SetInput("QKVW", {fuse_qkv_matmul_w_node->Name()});
    fused_attention_op_desc.SetInput("QKVBias",
                                     {fuse_qkv_ele_add_bias_node->Name()});
    fused_attention_op_desc.SetOutput("QKVOut",
                                      {fuse_qkv_matmul_out_node->Name()});
    fused_attention_op_desc.SetOutput("QKVBiasOut",
                                      {fuse_qkv_ele_add_out_node->Name()});
    fused_attention_op_desc.SetOutput("TransposeOut2",
                                      {fuse_qkv_transpose_out_node->Name()});

    GET_IR_NODE_FROM_SUBGRAPH(
        qk_matmul_out_node, qk_matmul_out, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(add_mask_ele_add_mask_node,
                              add_mask_ele_add_mask,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(add_mask_ele_add_out_node,
                              add_mask_ele_add_out,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_softmax_out_node, qk_softmax_out, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dropout_out_node, attn_dropout_out, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        attn_dropout_mask_node, attn_dropout_mask, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_matmul_out_node, qkv_matmul_out, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qkv_reshape_out_node, qkv_reshape_out, fused_attention_pattern);
    fused_attention_op_desc.SetOutput("QKOut", {qk_matmul_out_node->Name()});
    fused_attention_op_desc.SetInput("SrcMask",
                                     {add_mask_ele_add_mask_node->Name()});
    fused_attention_op_desc.SetOutput("SrcMaskOut",
                                      {add_mask_ele_add_out_node->Name()});
    fused_attention_op_desc.SetOutput("SoftmaxOut",
                                      {qk_softmax_out_node->Name()});
    fused_attention_op_desc.SetAttr(
        "attn_dropout_rate",
        PADDLE_GET_CONST(float,
                         attn_dropout_op_node->Op()->GetAttr("dropout_prob")));
    fused_attention_op_desc.SetAttr(
        "is_test",
        PADDLE_GET_CONST(bool, attn_dropout_op_node->Op()->GetAttr("is_test")));
    fused_attention_op_desc.SetAttr(
        "attn_dropout_fix_seed",
        PADDLE_GET_CONST(bool,
                         attn_dropout_op_node->Op()->GetAttr("fix_seed")));
    fused_attention_op_desc.SetAttr(
        "attn_dropout_seed",
        PADDLE_GET_CONST(int, attn_dropout_op_node->Op()->GetAttr("seed")));
    fused_attention_op_desc.SetAttr(
        "attn_dropout_implementation",
        PADDLE_GET_CONST(
            std::string,
            attn_dropout_op_node->Op()->GetAttr("dropout_implementation")));
    fused_attention_op_desc.SetOutput("AttnDropoutMaskOut",
                                      {attn_dropout_mask_node->Name()});
    fused_attention_op_desc.SetOutput("AttnDropoutOut",
                                      {attn_dropout_out_node->Name()});
    fused_attention_op_desc.SetOutput("QKTVOut", {qkv_matmul_out_node->Name()});
    fused_attention_op_desc.SetOutput("FMHAOut",
                                      {qkv_reshape_out_node->Name()});

    GET_IR_NODE_FROM_SUBGRAPH(
        out_linear_matmul_w_node, out_linear_matmul_w, fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_matmul_out_node,
                              out_linear_matmul_out,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_ele_add_bias_node,
                              out_linear_ele_add_bias,
                              fused_attention_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_ele_add_out_node,
                              out_linear_ele_add_out,
                              fused_attention_pattern);
    fused_attention_op_desc.SetInput("OutLinearW",
                                     {out_linear_matmul_w_node->Name()});
    fused_attention_op_desc.SetInput("OutLinearBias",
                                     {out_linear_ele_add_bias_node->Name()});
    fused_attention_op_desc.SetOutput("OutLinearOut",
                                      {out_linear_matmul_out_node->Name()});
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_dropout_mask_node,
                              out_linear_dropout_mask,
                              fused_attention_pattern);
    fused_attention_op_desc.SetAttr(
        "dropout_rate",
        PADDLE_GET_CONST(
            float, out_linear_dropout_op_node->Op()->GetAttr("dropout_prob")));
    fused_attention_op_desc.SetAttr(
        "dropout_fix_seed",
        PADDLE_GET_CONST(
            bool, out_linear_dropout_op_node->Op()->GetAttr("fix_seed")));
    fused_attention_op_desc.SetAttr(
        "dropout_seed",
        PADDLE_GET_CONST(int,
                         out_linear_dropout_op_node->Op()->GetAttr("seed")));
    fused_attention_op_desc.SetAttr(
        "dropout_implementation",
        PADDLE_GET_CONST(std::string,
                         out_linear_dropout_op_node->Op()->GetAttr(
                             "dropout_implementation")));
    fused_attention_op_desc.SetOutput("DropoutMaskOut",
                                      {out_linear_dropout_mask_node->Name()});

    GET_IR_NODE_FROM_SUBGRAPH(residual_ele_add_out_node,
                              residual_ele_add_out,
                              fused_attention_pattern);
    fused_attention_op_desc.SetAttr("add_residual", true);
    fused_attention_op_desc.SetOutput("Y", {residual_ele_add_out_node->Name()});

    auto fused_attention_node = g->CreateOpNode(&fused_attention_op_desc);

    IR_NODE_LINK_TO(subgraph.at(x), fused_attention_node);
    IR_NODE_LINK_TO(pre_layer_norm_scale_node, fused_attention_node);
    IR_NODE_LINK_TO(pre_layer_norm_bias_node, fused_attention_node);
    IR_NODE_LINK_TO(fuse_qkv_matmul_w_node, fused_attention_node);
    IR_NODE_LINK_TO(fuse_qkv_ele_add_bias_node, fused_attention_node);
    IR_NODE_LINK_TO(add_mask_ele_add_mask_node, fused_attention_node);
    IR_NODE_LINK_TO(out_linear_matmul_w_node, fused_attention_node);
    IR_NODE_LINK_TO(out_linear_ele_add_bias_node, fused_attention_node);

    IR_NODE_LINK_TO(fused_attention_node, pre_layer_norm_out_node);
    IR_NODE_LINK_TO(fused_attention_node, pre_layer_norm_mean_node);
    IR_NODE_LINK_TO(fused_attention_node, pre_layer_norm_variance_node);
    IR_NODE_LINK_TO(fused_attention_node, fuse_qkv_matmul_out_node);
    IR_NODE_LINK_TO(fused_attention_node, fuse_qkv_ele_add_out_node);
    IR_NODE_LINK_TO(fused_attention_node, fuse_qkv_transpose_out_node);
    IR_NODE_LINK_TO(fused_attention_node, qk_matmul_out_node);
    IR_NODE_LINK_TO(fused_attention_node, add_mask_ele_add_out_node);
    IR_NODE_LINK_TO(fused_attention_node, qk_softmax_out_node);
    IR_NODE_LINK_TO(fused_attention_node, attn_dropout_mask_node);
    IR_NODE_LINK_TO(fused_attention_node, attn_dropout_out_node);
    IR_NODE_LINK_TO(fused_attention_node, qkv_matmul_out_node);
    IR_NODE_LINK_TO(fused_attention_node, qkv_reshape_out_node);
    IR_NODE_LINK_TO(fused_attention_node, out_linear_matmul_out_node);
    IR_NODE_LINK_TO(fused_attention_node, out_linear_dropout_mask_node);
    IR_NODE_LINK_TO(fused_attention_node, residual_ele_add_out_node);

    GraphSafeRemoveNodes(g,
                         {pre_layer_norm_op_node,
                          fuse_qkv_matmul_op_node,
                          fuse_qkv_ele_add_op_node,
                          fuse_qkv_reshape_op_node,
                          fuse_qkv_transpose_op_node,
                          fuse_qkv_split_op_node,
                          qk_matmul_op_node,
                          qk_scale_op_node,
                          add_mask_ele_add_op_node,
                          qk_softmax_op_node,
                          attn_dropout_op_node,
                          qkv_matmul_op_node,
                          qkv_transpose_op_node,
                          qkv_reshape_op_node,
                          out_linear_matmul_op_node,
                          out_linear_ele_add_op_node,
                          out_linear_dropout_op_node,
                          residual_ele_add_op_node});
    found_fused_attention++;
  };

  gpd(graph, handler);
  AddStatis(found_fused_attention);

  return graph;
}

ir::Graph* FusedAttentionsPass::PreMaskDropResBwd(Graph* graph) const {
  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode(patterns::PDNodeName(name_scope_, "x"))
                ->AsInput()
                ->assert_is_op_input("elementwise_add_grad", "Out@GRAD");
  patterns::FusedAttentionGradPattern fused_attention_grad_pattern(
      gpd.mutable_pattern(), "fused_attention_grad_pattern");

  fused_attention_grad_pattern(x,
                               /* pre_layer_norm */ true,
                               /* has_attn_mask */ true,
                               /* do_dropout */ true,
                               /* add_residual */ true);

  int found_fused_attention = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "handle FusedMultiHeadAttention backward pass's fusion";

    GET_IR_NODE_FROM_SUBGRAPH(residual_ele_add_grad_op_node,
                              residual_ele_add_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_dropout_grad_op_node,
                              out_linear_dropout_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_ele_add_grad_op_node,
                              out_linear_ele_add_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out_linear_matmul_grad_op_node,
                              out_linear_matmul_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_reshape_grad_op_node,
                              qkv_reshape_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_transpose_grad_op_node,
                              qkv_transpose_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(qkv_matmul_grad_op_node,
                              qkv_matmul_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(attn_dropout_grad_op_node,
                              attn_dropout_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(qk_softmax_grad_op_node,
                              qk_softmax_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(add_mask_ele_add_grad_op_node,
                              add_mask_ele_add_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        qk_scale_grad_op_node, qk_scale_grad_op, fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(qk_matmul_grad_op_node,
                              qk_matmul_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_split_grad_op_node,
                              fuse_qkv_split_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_transpose_grad_op_node,
                              fuse_qkv_transpose_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_reshape_grad_op_node,
                              fuse_qkv_reshape_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_ele_add_grad_op_node,
                              fuse_qkv_ele_add_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fuse_qkv_matmul_grad_op_node,
                              fuse_qkv_matmul_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pre_layer_norm_grad_op_node,
                              pre_layer_norm_grad_op,
                              fused_attention_grad_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(grad_accumulation_sum_op_node,
                              grad_accumulation_sum_op,
                              fused_attention_grad_pattern);

    // TODO(Yuang Liu): finish the handler

    GraphSafeRemoveNodes(g,
                         {residual_ele_add_grad_op_node,
                          out_linear_dropout_grad_op_node,
                          out_linear_ele_add_grad_op_node,
                          out_linear_matmul_grad_op_node,
                          qkv_reshape_grad_op_node,
                          qkv_transpose_grad_op_node,
                          qkv_matmul_grad_op_node,
                          attn_dropout_grad_op_node,
                          qk_softmax_grad_op_node,
                          add_mask_ele_add_grad_op_node,
                          qk_scale_grad_op_node,
                          qk_matmul_grad_op_node,
                          fuse_qkv_split_grad_op_node,
                          fuse_qkv_transpose_grad_op_node,
                          fuse_qkv_reshape_grad_op_node,
                          fuse_qkv_ele_add_grad_op_node,
                          fuse_qkv_matmul_grad_op_node,
                          pre_layer_norm_grad_op_node,
                          grad_accumulation_sum_op_node});

    found_fused_attention++;
  };

  gpd(graph, handler);
  AddStatis(found_fused_attention);

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_attention_pass, paddle::framework::ir::FusedAttentionsPass);
