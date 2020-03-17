// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

struct EmbeddingEltwiseLayerNormPattern : public PatternBase {
  EmbeddingEltwiseLayerNormPattern(PDPattern* pattern,
                                   const std::string& name_scope)
      : PatternBase(pattern, name_scope, "embedding_eltwise_layernorm") {}

  PDNode* operator()();

  PATTERN_DECL_NODE(lookup_table1_x);
  PATTERN_DECL_NODE(lookup_table2_x);
  PATTERN_DECL_NODE(lookup_table3_x);

  PATTERN_DECL_NODE(lookup_table1_w);
  PATTERN_DECL_NODE(lookup_table2_w);
  PATTERN_DECL_NODE(lookup_table3_w);

  PATTERN_DECL_NODE(lookup_table1);
  PATTERN_DECL_NODE(lookup_table2);
  PATTERN_DECL_NODE(lookup_table3);

  PATTERN_DECL_NODE(lookup_table1_out);
  PATTERN_DECL_NODE(lookup_table2_out);
  PATTERN_DECL_NODE(lookup_table3_out);

  PATTERN_DECL_NODE(eltwise_add_12);
  PATTERN_DECL_NODE(eltwise_add_12_out);

  PATTERN_DECL_NODE(eltwise_add);
  PATTERN_DECL_NODE(eltwise_add_out);

  PATTERN_DECL_NODE(layer_norm);
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  // Delete the mean and var nodes in the graph.
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
};
}  // namespace patterns

// The EmbeddingEltwiseLayerNormFusePass detect the following pattern:
//
// inputs                           operator            output
// --------------------------------------------------------------------
// (word, weights_0)                lookup_table     ->  word_emb
// (pos, weights_1)                 lookup_table     ->  pos_emb
// (sent, weights_2)                lookup_table     ->  sent_emb
// (word_emb, pos_emb)              elementweise_add -> elementwise_out_0
// (elemtwise_out_0, sent_emb)      elementweise_add -> elementwise_out_1
// (elementwise_out_1, scale, bias) layer_norm       -> layer_norm_out
//
// and then convert the corresponding subgraph to:
//
// (word, pos, sent, weights_0, weights_1, weights_2,
//       scale, baias)   embedding_eltwise_layernorm -> layer_norm_out

class EmbeddingEltwiseLayerNormFusePass : public FusePassBase {
 public:
  virtual ~EmbeddingEltwiseLayerNormFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const;

  const std::string name_scope_{"embedding_eltwise_layernorm_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
