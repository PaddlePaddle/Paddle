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

#pragma once

#include <memory>
#include <string>
#include <utility>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct TrtPromptTuningEmbedding2Eltwise1Pattern : public PatternBase {
  TrtPromptTuningEmbedding2Eltwise1Pattern(PDPattern* pattern,
                                           const std::string& name_scope)
      : PatternBase(pattern, name_scope, "embedding2_eltwise1") {}

  void operator()();
  PATTERN_DECL_NODE(feed1);
  PATTERN_DECL_NODE(feed2);
  PATTERN_DECL_NODE(lookup_table1_x);
  PATTERN_DECL_NODE(lookup_table2_x);
  PATTERN_DECL_NODE(lookup_table1_w);
  PATTERN_DECL_NODE(lookup_table2_w);
  PATTERN_DECL_NODE(lookup_table1);
  PATTERN_DECL_NODE(lookup_table2);
  PATTERN_DECL_NODE(lookup_table1_out);
  PATTERN_DECL_NODE(lookup_table2_out);
  PATTERN_DECL_NODE(eltwise_add);
  PATTERN_DECL_NODE(eltwise_add_out);
};

struct TrtPromptTuningEmbedding1Eltwise1Pattern : public PatternBase {
  TrtPromptTuningEmbedding1Eltwise1Pattern(PDPattern* pattern,
                                           const std::string& name_scope)
      : PatternBase(pattern, name_scope, "embedding1_eltwise1") {}
  void operator()();
  PATTERN_DECL_NODE(feed1);
  PATTERN_DECL_NODE(lookup_table1_x);
  PATTERN_DECL_NODE(lookup_table1_w);
  PATTERN_DECL_NODE(lookup_table1);
  PATTERN_DECL_NODE(lookup_table1_out);
  PATTERN_DECL_NODE(eltwise_add_in);
  PATTERN_DECL_NODE(eltwise_add);
  PATTERN_DECL_NODE(eltwise_add_out);
};

struct TrtPromptTuningSkipLayerNorm : public PatternBase {
  TrtPromptTuningSkipLayerNorm(PDPattern* pattern,
                               const std::string& name_scope)
      : PatternBase(pattern, name_scope, "skip_layernorm") {}
  void operator()();

  PATTERN_DECL_NODE(eltwise_add);
  PATTERN_DECL_NODE(eltwise_add_out);
  PATTERN_DECL_NODE(mul0_x);
  PATTERN_DECL_NODE(mul0_y);
  PATTERN_DECL_NODE(mul0);
  PATTERN_DECL_NODE(mul0_out);
  PATTERN_DECL_NODE(eltadd0_b);
  PATTERN_DECL_NODE(eltadd0);
  PATTERN_DECL_NODE(eltadd0_out);
  PATTERN_DECL_NODE(relu);
  PATTERN_DECL_NODE(relu_out);
  PATTERN_DECL_NODE(mul1_y);
  PATTERN_DECL_NODE(mul1);
  PATTERN_DECL_NODE(mul1_out);
  PATTERN_DECL_NODE(eltadd1_b);
  PATTERN_DECL_NODE(eltadd1);
  PATTERN_DECL_NODE(eltadd1_out);
  PATTERN_DECL_NODE(concat);
  PATTERN_DECL_NODE(concat_out);
  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
};
}  // namespace patterns

class TrtPromptTuningEmbeddingEltwiseLayerNormFusePass : public FusePassBase {
 public:
  TrtPromptTuningEmbeddingEltwiseLayerNormFusePass();
  virtual ~TrtPromptTuningEmbeddingEltwiseLayerNormFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const;
  int BuildFusion(Graph* graph, const std::string& name_scope
                  /*const Scope* scope*/) const;
  const std::string name_scope_{
      "trt_prompt_tuning_embedding_eltwise_layernorm_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
