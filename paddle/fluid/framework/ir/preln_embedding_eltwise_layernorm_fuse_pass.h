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

// detect start pattern.
//
//     in_var  emb       in_var   emb
//       |      |          |       |
//     lookup_table      lookup_table
//           |                 |
//        lkt_var           lkt_var
//            \                /
//             elementwise_add
//                    |
//               elt_out_var
//
struct PrelnEmbedding2Eltwise1Pattern : public PatternBase {
  PrelnEmbedding2Eltwise1Pattern(PDPattern* pattern,
                                 const std::string& name_scope)
      : PatternBase(pattern, name_scope, "Prelnembedding2_eltwise1") {}

  void operator()();

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

// detect repeats inner pattern
//
//    elt_out_var            in_var   emb
//         \                   |       |
//          \                 lookup_table
//           \                     |
//            \                 lkt_var
//             \                   /
//                elementwise_add
//                  |        |
//        elementwise_add  elt_out_var
//
struct PrelnEmbedding1Eltwise1Pattern : public PatternBase {
  PrelnEmbedding1Eltwise1Pattern(PDPattern* pattern,
                                 const std::string& name_scope)
      : PatternBase(pattern, name_scope, "Prelnembedding1_eltwise1") {}
  void operator()();
  PATTERN_DECL_NODE(lookup_table1_x);
  PATTERN_DECL_NODE(lookup_table1_w);
  PATTERN_DECL_NODE(lookup_table1);
  PATTERN_DECL_NODE(lookup_table1_out);
  PATTERN_DECL_NODE(eltwise_add_in);
  PATTERN_DECL_NODE(eltwise_add);
  PATTERN_DECL_NODE(eltwise_add_out);
};

// detect end pattern
//
//     elementwise_add
//      |          |
//      |       elt_out_var
//      |      scale   |   bias
//      |           \  |  /
// elementwise_add  layer_norm
//
struct PrelnSkipLayerNorm : public PatternBase {
  PrelnSkipLayerNorm(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "Prelnskip_layernorm") {}
  void operator()();
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

// The PrelnEmbeddingEltwiseLayerNormFusePass detect the following pattern:
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
//       scale, baias)   Prelnembedding_eltwise_layernorm -> layer_norm_out +
//       elementwise_add_out
//
//
//  in_var  emb_var   in_var   emb_var   in_var   emb_var      in_var   emb_var
//    |        |        |         |        |         |           |         |
//   lookup_table      lookup_table       lookup_table   ...    lookup_table
//        |                 |                  |                     |
//     lkt_var           lkt_var            lkt_var               lkt_var
//        \                 /                  |         ...         |
//          elementwise_add                    |                     |
//                 \                          /                      |
//                       elementwise_add                             |
//                               |                                   |
//                            elt_var                               /
//                               \                                 /
//                                         elementwise_add
//                                           |         |
//                                elementwise_add    layer_norm

class PrelnEmbeddingEltwiseLayerNormFusePass : public FusePassBase {
 public:
  PrelnEmbeddingEltwiseLayerNormFusePass();
  virtual ~PrelnEmbeddingEltwiseLayerNormFusePass() {}

 protected:
  void ApplyImpl(Graph* graph) const;
  int BuildFusion(Graph* graph, const std::string& name_scope
                  /*const Scope* scope*/) const;
  const std::string name_scope_{"preln_embedding_eltwise_layernorm_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
