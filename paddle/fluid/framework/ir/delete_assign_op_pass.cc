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

struct AssignWithSameInputOutputNamePattern : public PatternBase {
  AssignWithSameInputOutputNamePattern(PDPattern* pattern,
                                       const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(assign);
};

AssignWithSameInputOutputNamePattern::AssignWithSameInputOutputNamePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  pattern->NewNode(assign_repr())
      ->assert_is_op("assign")
      ->assert_more([](Node* node) {
        auto in_name = node->Op()->Input("X")[0];
        auto out_name = node->Op()->Output("Out")[0];
        return in_name == out_name;
      });
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

/*
Delete "assign" if its input and output is same.
*/
class DeleteAssignOpPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"delete_assign_op_pass"};
};

void DeleteAssignOpPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::AssignWithSameInputOutputNamePattern pattern(gpd.mutable_pattern(),
                                                         name_scope_);

  int found_subgraph_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DeleteAssignOpPass fuse";
    GET_IR_NODE_FROM_SUBGRAPH(assign, assign, pattern);

    std::unordered_set<const Node*> delete_nodes{assign};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_assign_op_pass, paddle::framework::ir::DeleteAssignOpPass);

REGISTER_PASS_CAPABILITY(delete_assign_op_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "assign", 0));
