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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::framework::ir::patterns {

struct FillMulPattern : public PatternBase {
  FillMulPattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fill);
  PATTERN_DECL_NODE(mul);
  // declare variable node's name
  PATTERN_DECL_NODE(fill_out);
  PATTERN_DECL_NODE(mul_in);
  PATTERN_DECL_NODE(mul_out);
};

FillMulPattern::FillMulPattern(PDPattern* pattern,
                               const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* fill = pattern->NewNode(fill_repr())
                   ->assert_is_op("fill_constant_batch_size_like")
                   ->assert_more([](Node* node) {
                     float value = node->Op()->GetAttrIfExists<float>("value");
                     return fabs(value - 1.f) < 1e-5;
                   });
  auto* fill_out =
      pattern->NewNode(fill_out_repr())
          ->assert_is_op_output("fill_constant_batch_size_like", "Out")
          ->assert_has_n_outputs(1);
  auto* mul_in = pattern->NewNode(mul_in_repr());
  auto* mul = pattern->NewNode(mul_repr())->assert_is_op("elementwise_mul");
  auto* mul_out = pattern->NewNode(mul_out_repr())
                      ->assert_is_op_output("elementwise_mul", "Out");

  fill->LinksTo({fill_out});
  mul->LinksFrom({fill_out, mul_in}).LinksTo({mul_out});
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

/*
Delete "elementwise" if one of inputs is "1".
*/
class DeleteElementwiseMulOpPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"delete_elementwise_mul_op_pass"};
};

void DeleteElementwiseMulOpPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::FillMulPattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteElementwiseMulOpPass fuse";
#define GET_IR_NODE(node_) GET_IR_NODE_FROM_SUBGRAPH(node_, node_, pattern)
    GET_IR_NODE(fill);
    GET_IR_NODE(mul);
    GET_IR_NODE(fill_out);
    GET_IR_NODE(mul_in);
    GET_IR_NODE(mul_out);
#undef GET_IR_NODE

    for (auto* next_op : mul_out->outputs) {
      next_op->Op()->RenameInput(mul_out->Name(), mul_in->Name());
      IR_NODE_LINK_TO(mul_in, next_op);
    }

    std::unordered_set<const Node*> delete_nodes{fill, mul, fill_out, mul_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_elementwise_mul_op_pass,
              paddle::framework::ir::DeleteElementwiseMulOpPass);

REGISTER_PASS_CAPABILITY(delete_elementwise_mul_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "fill_constant_batch_size_like", 0));
