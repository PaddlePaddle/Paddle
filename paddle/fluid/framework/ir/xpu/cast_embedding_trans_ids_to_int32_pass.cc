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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct CastEmbeddingTransIdsToInt32Pattern : public PatternBase {
  CastEmbeddingTransIdsToInt32Pattern(PDPattern* pattern,
                                      const std::string& name_scope);
  // declare operator node's name
  PATTERN_DECL_NODE(cast);
  PATTERN_DECL_NODE(embedding);
  // declare variable node's name
  PATTERN_DECL_NODE(cast_x);
  PATTERN_DECL_NODE(embedding_ids);
  PATTERN_DECL_NODE(embedding_w);
  PATTERN_DECL_NODE(embedding_out);
};

CastEmbeddingTransIdsToInt32Pattern::CastEmbeddingTransIdsToInt32Pattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto cast = pattern->NewNode(cast_repr())->assert_is_op("cast");
  auto cast_x = pattern->NewNode(cast_x_repr())
                    ->assert_is_op_input("cast", "X")
                    ->assert_var_not_persistable()
                    ->AsInput();
  auto embedding_ids = pattern->NewNode(embedding_ids_repr())
                           ->assert_is_op_output("cast", "Out")
                           ->assert_is_op_input("lookup_table_v2", "Ids")
                           ->assert_has_n_outputs(1);
  cast->LinksFrom({cast_x}).LinksTo({embedding_ids});
  auto embedding_w = pattern->NewNode(embedding_w_repr())
                         ->assert_is_op_input("lookup_table_v2", "W");
  auto embedding =
      pattern->NewNode(embedding_repr())->assert_is_op("lookup_table_v2");
  auto embedding_out = pattern->NewNode(embedding_out_repr())
                           ->assert_is_op_output("lookup_table_v2", "Out")
                           ->AsOutput();
  embedding->LinksFrom({embedding_ids, embedding_w}).LinksTo({embedding_out});
}

}  // namespace patterns

class CastEmbeddingTransIdsToInt32Pass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"cast_embedding_trans_ids_to_int32_pass"};
};
void CastEmbeddingTransIdsToInt32Pass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  GraphPatternDetector gpd;
  patterns::CastEmbeddingTransIdsToInt32Pattern pattern(gpd.mutable_pattern(),
                                                        name_scope_);
  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle CastEmbeddingTransIdsToInt32Pass";
    GET_IR_NODE(cast);
    GET_IR_NODE(embedding);
    GET_IR_NODE(embedding_ids);
    auto cast_node_attr_out_dtype =
        cast->Op()->GetAttrIfExists<int>("out_dtype");
    if (cast_node_attr_out_dtype !=
        static_cast<int>(paddle::framework::proto::VarType::INT64)) {
      return;
    }
    cast->Op()->SetAttr(
        "out_dtype",
        static_cast<int>(paddle::framework::proto::VarType::INT32));
    embedding_ids->Var()->SetDataType(paddle::framework::proto::VarType::INT32);
    embedding->Op()->Flush();
    found_subgraph_count++;
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
  if (found_subgraph_count) {
    VLOG(4) << "There is a risk of overflow when converting the data type of "
               "embedded ids from int64 to int32."
               "Please ensure that the numerical range of ids is within the "
               "maximum value of int32."
               "If it exceeds this range, it may result in incorrect results. "
               "You can try removing this pass.";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cast_embedding_trans_ids_to_int32_pass,
              paddle::framework::ir::CastEmbeddingTransIdsToInt32Pass);

REGISTER_PASS_CAPABILITY(cast_embedding_trans_ids_to_int32_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "lookup_table_v2", 1));
