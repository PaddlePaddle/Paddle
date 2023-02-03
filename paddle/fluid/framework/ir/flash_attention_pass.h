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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FlashAttentionPattern : public PatternBase {
  FlashAttentionPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "flash_attention_pattern") {}

  PDNode* operator()(PDNode* x,
                     bool scale_before_matmul,
                     bool is_causal,
                     bool is_dropout);

  PATTERN_DECL_NODE(q_transpose_op);
  PATTERN_DECL_NODE(q_transpose_out);
  PATTERN_DECL_NODE(q_transpose_xshape);

  PATTERN_DECL_NODE(k_transpose_in);
  PATTERN_DECL_NODE(k_transpose_op);
  PATTERN_DECL_NODE(k_transpose_out);
  PATTERN_DECL_NODE(k_transpose_xshape);

  PATTERN_DECL_NODE(v_transpose_in);
  PATTERN_DECL_NODE(v_transpose_op);
  PATTERN_DECL_NODE(v_transpose_out);
  PATTERN_DECL_NODE(v_transpose_xshape);

  PATTERN_DECL_NODE(scale_op);
  PATTERN_DECL_NODE(scale_out);

  PATTERN_DECL_NODE(qk_matmul_op);
  PATTERN_DECL_NODE(qk_matmul_out);

  PATTERN_DECL_NODE(qk_softmax_op);
  PATTERN_DECL_NODE(qk_softmax_out);

  PATTERN_DECL_NODE(dropout_op);
  PATTERN_DECL_NODE(dropout_out);
  PATTERN_DECL_NODE(dropout_mask);

  PATTERN_DECL_NODE(qkv_matmul_op);
  PATTERN_DECL_NODE(qkv_matmul_out);

  PATTERN_DECL_NODE(qkv_transpose_op);
  PATTERN_DECL_NODE(qkv_transpose_out);
  PATTERN_DECL_NODE(qkv_transpose_xshape);
};

// Declare the grad pattern for multi head attention
struct FlashAttentionGradPattern : public PatternBase {
  FlashAttentionGradPattern(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "flash_attention_pattern") {}

  PDNode* operator()(PDNode* x,
                     bool scale_before_matmul,
                     bool is_causal,
                     bool is_dropout);
};

}  // namespace patterns

class FlashAttentionsPass : public FusePassBase {
 public:
  virtual ~FlashAttentionsPass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"flash_attention_pass"};

 private:
  ir::Graph* FlashAttentionFwd(Graph* graph) const;

  ir::Graph* FlashAttentionBwd(Graph* graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
