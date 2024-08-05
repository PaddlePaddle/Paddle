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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

// Declare patterns for multi head attention.
// Can detect:
// 1. Pre layer norm or post layer norm.
// 2. Add attn mask for qk product before the softmax or not.
// 3. Do attn dropout or not.
// 4. Add residual to the out linear result or not.
// 5. Use model tensor parallel or not.
struct FusedAttentionPattern : public PatternBase {
  FusedAttentionPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fused_attention_pattern") {}

  PDNode* operator()(PDNode* x,
                     bool pre_layer_norm,  // do pre ln or not
                     bool has_attn_mask,   // add attn mask to qk or not
                     bool do_dropout,      // dropout the softmax(qk) or not
                     bool add_residual,    // add residual to out linear or not
                     bool use_mp);         // use tensor parallel or not

  // pre layer norm
  PATTERN_DECL_NODE(pre_layer_norm_op);
  PATTERN_DECL_NODE(pre_layer_norm_scale);
  PATTERN_DECL_NODE(pre_layer_norm_bias);
  PATTERN_DECL_NODE(pre_layer_norm_out);
  PATTERN_DECL_NODE(pre_layer_norm_mean);
  PATTERN_DECL_NODE(pre_layer_norm_variance);

  // c_identity for mp
  PATTERN_DECL_NODE(c_identity_op);
  PATTERN_DECL_NODE(c_identity_out);

  // fuse qkv projection
  PATTERN_DECL_NODE(fuse_qkv_matmul_op);
  PATTERN_DECL_NODE(fuse_qkv_matmul_w);
  PATTERN_DECL_NODE(fuse_qkv_matmul_out);

  PATTERN_DECL_NODE(fuse_qkv_ele_add_op);
  PATTERN_DECL_NODE(fuse_qkv_ele_add_bias);
  PATTERN_DECL_NODE(fuse_qkv_ele_add_out);

  PATTERN_DECL_NODE(fuse_qkv_reshape_op);
  PATTERN_DECL_NODE(fuse_qkv_reshape_out);
  PATTERN_DECL_NODE(fuse_qkv_reshape_x_shape);

  PATTERN_DECL_NODE(fuse_qkv_transpose_op);
  PATTERN_DECL_NODE(fuse_qkv_transpose_out);
  PATTERN_DECL_NODE(fuse_qkv_transpose_x_shape);

  PATTERN_DECL_NODE(fuse_qkv_split_op);
  PATTERN_DECL_NODE(fuse_qkv_split_out_q);  // q
  PATTERN_DECL_NODE(fuse_qkv_split_out_k);  // k
  PATTERN_DECL_NODE(fuse_qkv_split_out_v);  // v

  // core attention
  PATTERN_DECL_NODE(qk_matmul_op);
  PATTERN_DECL_NODE(qk_matmul_out);

  PATTERN_DECL_NODE(qk_scale_op);
  PATTERN_DECL_NODE(qk_scale_out);

  PATTERN_DECL_NODE(add_mask_ele_add_op);
  PATTERN_DECL_NODE(add_mask_ele_add_mask);
  PATTERN_DECL_NODE(add_mask_ele_add_out);

  PATTERN_DECL_NODE(qk_softmax_op);
  PATTERN_DECL_NODE(qk_softmax_out);

  PATTERN_DECL_NODE(attn_dropout_op);
  PATTERN_DECL_NODE(attn_dropout_out);
  PATTERN_DECL_NODE(attn_dropout_mask);

  PATTERN_DECL_NODE(qkv_matmul_op);
  PATTERN_DECL_NODE(qkv_matmul_out);

  PATTERN_DECL_NODE(qkv_transpose_op);
  PATTERN_DECL_NODE(qkv_transpose_out);
  PATTERN_DECL_NODE(qkv_transpose_x_shape);

  PATTERN_DECL_NODE(qkv_reshape_op);
  PATTERN_DECL_NODE(qkv_reshape_out);
  PATTERN_DECL_NODE(qkv_reshape_x_shape);

  // out linear
  PATTERN_DECL_NODE(out_linear_matmul_op);
  PATTERN_DECL_NODE(out_linear_matmul_w);
  PATTERN_DECL_NODE(out_linear_matmul_out);

  PATTERN_DECL_NODE(out_linear_ele_add_op);
  PATTERN_DECL_NODE(out_linear_ele_add_bias);
  PATTERN_DECL_NODE(out_linear_ele_add_out);

  // allreduce for mp
  PATTERN_DECL_NODE(mp_allreduce_sum_op);
  PATTERN_DECL_NODE(mp_allreduce_sum_out);

  PATTERN_DECL_NODE(out_linear_dropout_op);
  PATTERN_DECL_NODE(out_linear_dropout_out);
  PATTERN_DECL_NODE(out_linear_dropout_mask);

  // residual
  PATTERN_DECL_NODE(residual_ele_add_op);
  PATTERN_DECL_NODE(residual_ele_add_out);

  // post layer norm
  PATTERN_DECL_NODE(post_layer_norm_op);
  PATTERN_DECL_NODE(post_layer_norm_scale);
  PATTERN_DECL_NODE(post_layer_norm_bias);
  PATTERN_DECL_NODE(post_layer_norm_out);
  PATTERN_DECL_NODE(post_layer_norm_mean);
  PATTERN_DECL_NODE(post_layer_norm_variance);
};

// Declare the grad pattern for multi head attention
struct FusedAttentionGradPattern : public PatternBase {
  FusedAttentionGradPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "fused_attention_grad_pattern") {}

  PDNode* operator()(PDNode* x,
                     bool pre_layer_norm,  // pre ln
                     bool has_attn_mask,   // add attn mask to qk or not
                     bool do_dropout,      // dropout the softmax(qk) or not
                     bool add_residual,    // add residual to out linear or not
                     bool use_mp);         // use tensor parallel or not

  // post layer norm grad
  PATTERN_DECL_NODE(post_layer_norm_grad_op);
  PATTERN_DECL_NODE(post_layer_norm_grad_scale);
  PATTERN_DECL_NODE(post_layer_norm_grad_bias);
  PATTERN_DECL_NODE(post_layer_norm_grad_mean);
  PATTERN_DECL_NODE(post_layer_norm_grad_variance);
  PATTERN_DECL_NODE(post_layer_norm_grad_x);
  PATTERN_DECL_NODE(post_layer_norm_grad_scale_grad);
  PATTERN_DECL_NODE(post_layer_norm_grad_bias_grad);
  PATTERN_DECL_NODE(post_layer_norm_grad_x_grad);

  // residual grad
  PATTERN_DECL_NODE(residual_ele_add_grad_op);
  PATTERN_DECL_NODE(residual_ele_add_grad_x);
  PATTERN_DECL_NODE(residual_ele_add_grad_bias);
  PATTERN_DECL_NODE(residual_ele_add_grad_bias_grad);
  PATTERN_DECL_NODE(residual_ele_add_grad_x_grad);

  // out linear grad
  PATTERN_DECL_NODE(out_linear_dropout_grad_op);
  PATTERN_DECL_NODE(out_linear_dropout_grad_mask);
  PATTERN_DECL_NODE(out_linear_dropout_grad_out);

  // c_identity for mp
  PATTERN_DECL_NODE(mp_allreduce_sum_grad_op);  // c_identity
  PATTERN_DECL_NODE(mp_allreduce_sum_grad_out);

  PATTERN_DECL_NODE(out_linear_ele_add_grad_op);
  PATTERN_DECL_NODE(out_linear_ele_add_grad_x);
  PATTERN_DECL_NODE(out_linear_ele_add_grad_bias);
  PATTERN_DECL_NODE(out_linear_ele_add_grad_x_grad);
  PATTERN_DECL_NODE(out_linear_ele_add_grad_bias_grad);

  PATTERN_DECL_NODE(out_linear_matmul_grad_op);
  PATTERN_DECL_NODE(out_linear_matmul_grad_x);
  PATTERN_DECL_NODE(out_linear_matmul_grad_w);
  PATTERN_DECL_NODE(out_linear_matmul_grad_x_grad);
  PATTERN_DECL_NODE(out_linear_matmul_grad_w_grad);

  // core attention grad
  PATTERN_DECL_NODE(qkv_reshape_grad_op);
  PATTERN_DECL_NODE(qkv_reshape_grad_x_shape);
  PATTERN_DECL_NODE(qkv_reshape_grad_out);

  PATTERN_DECL_NODE(qkv_transpose_grad_op);
  PATTERN_DECL_NODE(qkv_transpose_grad_x_shape);
  PATTERN_DECL_NODE(qkv_transpose_grad_out);

  PATTERN_DECL_NODE(qkv_matmul_grad_op);
  PATTERN_DECL_NODE(qkv_matmul_grad_x);
  PATTERN_DECL_NODE(qkv_matmul_grad_w);
  PATTERN_DECL_NODE(qkv_matmul_grad_x_grad);
  PATTERN_DECL_NODE(qkv_matmul_grad_w_grad);

  PATTERN_DECL_NODE(attn_dropout_grad_op);
  PATTERN_DECL_NODE(attn_dropout_grad_mask);
  PATTERN_DECL_NODE(attn_dropout_grad_out);

  PATTERN_DECL_NODE(qk_softmax_grad_op);
  PATTERN_DECL_NODE(qk_softmax_grad_fwd_out);
  PATTERN_DECL_NODE(qk_softmax_grad_out);

  PATTERN_DECL_NODE(add_mask_ele_add_grad_op);
  PATTERN_DECL_NODE(add_mask_ele_add_grad_x);
  PATTERN_DECL_NODE(add_mask_ele_add_grad_bias);
  PATTERN_DECL_NODE(add_mask_ele_add_grad_x_grad);

  PATTERN_DECL_NODE(qk_scale_grad_op);
  PATTERN_DECL_NODE(qk_scale_grad_out);

  PATTERN_DECL_NODE(qk_matmul_grad_op);
  PATTERN_DECL_NODE(qk_matmul_grad_x);
  PATTERN_DECL_NODE(qk_matmul_grad_w);
  PATTERN_DECL_NODE(qk_matmul_grad_x_grad);
  PATTERN_DECL_NODE(qk_matmul_grad_w_grad);

  // fuse qkv projection grad
  PATTERN_DECL_NODE(fuse_qkv_split_grad_op);  // concat op
  PATTERN_DECL_NODE(fuse_qkv_split_grad_out);

  PATTERN_DECL_NODE(fuse_qkv_transpose_grad_op);
  PATTERN_DECL_NODE(fuse_qkv_transpose_grad_x_shape);
  PATTERN_DECL_NODE(fuse_qkv_transpose_grad_out);

  PATTERN_DECL_NODE(fuse_qkv_reshape_grad_op);
  PATTERN_DECL_NODE(fuse_qkv_reshape_grad_x_shape);
  PATTERN_DECL_NODE(fuse_qkv_reshape_grad_out);

  PATTERN_DECL_NODE(fuse_qkv_ele_add_grad_op);
  PATTERN_DECL_NODE(fuse_qkv_ele_add_grad_x);
  PATTERN_DECL_NODE(fuse_qkv_ele_add_grad_bias);
  PATTERN_DECL_NODE(fuse_qkv_ele_add_grad_x_grad);
  PATTERN_DECL_NODE(fuse_qkv_ele_add_grad_bias_grad);

  PATTERN_DECL_NODE(fuse_qkv_matmul_grad_op);
  PATTERN_DECL_NODE(fuse_qkv_matmul_grad_x);
  PATTERN_DECL_NODE(fuse_qkv_matmul_grad_w);
  PATTERN_DECL_NODE(fuse_qkv_matmul_grad_x_grad);
  PATTERN_DECL_NODE(fuse_qkv_matmul_grad_w_grad);

  // allreduce for mp
  PATTERN_DECL_NODE(c_identity_grad_op);  // mp_allreduce_sum
  PATTERN_DECL_NODE(c_identity_grad_out);

  // pre layer norm grad
  PATTERN_DECL_NODE(pre_layer_norm_grad_op);
  PATTERN_DECL_NODE(pre_layer_norm_grad_scale);
  PATTERN_DECL_NODE(pre_layer_norm_grad_bias);
  PATTERN_DECL_NODE(pre_layer_norm_grad_mean);
  PATTERN_DECL_NODE(pre_layer_norm_grad_variance);
  PATTERN_DECL_NODE(pre_layer_norm_grad_x);
  PATTERN_DECL_NODE(pre_layer_norm_grad_scale_grad);
  PATTERN_DECL_NODE(pre_layer_norm_grad_bias_grad);
  PATTERN_DECL_NODE(pre_layer_norm_grad_x_grad);

  // grad accumulation
  PATTERN_DECL_NODE(grad_accumulation_sum_op);
  PATTERN_DECL_NODE(grad_accumulation_out);
};

}  // namespace patterns

class FusedAttentionPassCache {
 public:
  ir::Node* GetNodeFromCache(const std::string name) {
    if (var_name_to_ir_node_cache_.count(name)) {
      return var_name_to_ir_node_cache_.find(name)->second;
    }
    PADDLE_THROW(common::errors::InvalidArgument(
        "The key (%d) of FusedAttentionCache does not exist.", name));
  }

  void InsertIntoCache(const std::string name, ir::Node* node) {
    if (!var_name_to_ir_node_cache_.count(name)) {
      var_name_to_ir_node_cache_.insert({name, node});
    } else {
      PADDLE_THROW(common::errors::AlreadyExists(
          "The key (%d) of FusedAttentionCache already exist.", name));
    }
  }

  void ResetCache() { var_name_to_ir_node_cache_.clear(); }

 private:
  std::unordered_map<std::string, ir::Node*> var_name_to_ir_node_cache_;
};

class FusedAttentionsPass : public FusePassBase {
 public:
  virtual ~FusedAttentionsPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"fused_attention_pass"};

 private:
  // The name rule for the helper function.
  // The function name will contain at most five parts in order:
  // 1. Do pre layer norm? [Pre]
  // 2. Add mask in the core attention part? [Mask]
  // 3. Do dropout in the core attention part? [Drop]
  // 4. Add residual? [Res]
  // 5. Do post layer norm? [Post]
  // 6. Forward or Backward? [Fwd/Bwd]
  // 7. Use tensor model parallel? [MP]
  // If true, the function name will have an abbreviation part.
  // If false, the function name won't contain an abbreviation for it.

  ir::Graph* PreMaskDropResFwd(Graph* graph,
                               FusedAttentionPassCache* cache) const;

  ir::Graph* PreMaskDropResBwd(Graph* graph,
                               FusedAttentionPassCache* cache) const;

  ir::Graph* PreMaskDropResMPFwd(Graph* graph,
                                 FusedAttentionPassCache* cache) const;

  ir::Graph* PreMaskDropResMPBwd(Graph* graph,
                                 FusedAttentionPassCache* cache) const;

  ir::Graph* ForwardHandlerHelper(Graph* graph,
                                  FusedAttentionPassCache* cache,
                                  bool pre_layer_norm,
                                  bool has_attn_mask,
                                  bool do_dropout,
                                  bool add_residual,
                                  bool use_mp) const;

  ir::Graph* BackwardHandlerHelper(Graph* graph,
                                   FusedAttentionPassCache* cache,
                                   bool pre_layer_norm,
                                   bool has_attn_mask,
                                   bool do_dropout,
                                   bool add_residual,
                                   bool use_mp) const;

  const std::string GenerateCacheKey(const std::string anchor,
                                     const std::string var_name,
                                     int block_id) const {
    return anchor + "_" + std::to_string(block_id) + "_" + var_name;
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
