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

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct ContinuousSameOpsPattern : public PatternBase {
  ContinuousSameOpsPattern(PDPattern* pattern,
                           const std::string& name_scope,
                           const std::string& op_type);
  PATTERN_DECL_NODE(first_in_var_node);
  PATTERN_DECL_NODE(first_out_var_node);
  PATTERN_DECL_NODE(second_out_var_node);
  // declare op node's name
  PATTERN_DECL_NODE(first_op_node);
  PATTERN_DECL_NODE(second_op_node);
  std::string op_type_;
};

ContinuousSameOpsPattern::ContinuousSameOpsPattern(
    PDPattern* pattern,
    const std::string& name_scope,
    const std::string& op_type)
    : PatternBase(pattern, name_scope, name_scope), op_type_(op_type) {
  auto* first_in_var_node =
      pattern->NewNode(first_in_var_node_repr())
          ->assert_var_not_persistable()
          ->assert_is_op_input(op_type_, "X")
          ->AsInput()
          ->assert_more([&](Node* node) {
            // assert pre op type is not same.
            auto input_nodes = node->inputs;
            if (input_nodes.size() != 1) return false;
            if (!input_nodes.empty() && input_nodes[0]->IsOp() &&
                input_nodes[0]->Op()->Type() == op_type_) {
              return false;
            }
            return true;
          });
  auto* first_op_node =
      pattern->NewNode(first_op_node_repr())->assert_is_op(op_type_);
  auto* first_out_var_node = pattern->NewNode(first_out_var_node_repr())
                                 ->assert_var_not_persistable()
                                 ->assert_is_op_output(op_type_, "Out")
                                 ->assert_has_n_outputs(1);
  first_op_node->LinksFrom({first_in_var_node}).LinksTo({first_out_var_node});
  auto* second_op_node =
      pattern->NewNode(second_op_node_repr())->assert_is_op(op_type_);
  auto* second_out_var_node = pattern->NewNode(second_out_var_node_repr())
                                  ->assert_var_not_persistable()
                                  ->assert_is_op_output(op_type_, "Out")
                                  ->AsOutput();
  second_op_node->LinksFrom({first_out_var_node})
      .LinksTo({second_out_var_node});
}

}  // namespace patterns

/*
Fused continuous same ops into one.
Origin graph:
    input
      |
      |
  unsqueeze2
      |
      |
  unsqueeze2
      |
      |
  unsqueeze2
      |
      |
     out

After:
    input
      |
      |
  unsqueeze2
      |
      |
     out
*/

class FusedContinuousSameOpsPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void FusedReshapeOps(ir::Graph* graph) const;
  void FusedUnsqueezeOps(ir::Graph* graph) const;

  const std::string name_scope_{"fused_continuous_same_ops_pass"};
  mutable int delete_op_count{0};
};

void FusedContinuousSameOpsPass::FusedReshapeOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ContinuousSameOpsPattern pattern(
      gpd.mutable_pattern(), name_scope_, "reshape2");
  int delete_counts = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle fused continuous reshape ops.";
    GET_IR_NODE_FROM_SUBGRAPH(first_in_var_node, first_in_var_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(first_out_var_node, first_out_var_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        second_out_var_node, second_out_var_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(first_op_node, first_op_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(second_op_node, second_op_node, pattern);
    auto first_node_attr_shape =
        first_op_node->Op()->GetAttrIfExists<std::vector<int>>("shape");
    if (first_node_attr_shape.empty()) return;
    auto second_node_attr_shape =
        second_op_node->Op()->GetAttrIfExists<std::vector<int>>("shape");
    if (second_node_attr_shape.empty()) return;
    second_op_node->Op()->RenameInput(first_out_var_node->Name(),
                                      first_in_var_node->Name());
    IR_NODE_LINK_TO(first_in_var_node, second_op_node);
    GraphSafeRemoveNodes(graph, {first_op_node, first_out_var_node});
    delete_counts++;
  };
  gpd(graph, handler);
  delete_op_count += delete_counts;
  if (delete_counts > 0) {
    LOG(INFO) << "--- delete " << delete_counts << " repeated "
              << "reshape2"
              << " ops";
  }
}
void FusedContinuousSameOpsPass::FusedUnsqueezeOps(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::ContinuousSameOpsPattern pattern(
      gpd.mutable_pattern(), name_scope_, "unsqueeze2");
  int delete_counts = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle fused continuous unsqueeze ops.";
    GET_IR_NODE_FROM_SUBGRAPH(first_in_var_node, first_in_var_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(first_out_var_node, first_out_var_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        second_out_var_node, second_out_var_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(first_op_node, first_op_node, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(second_op_node, second_op_node, pattern);
    auto first_node_attr_axes =
        first_op_node->Op()->GetAttrIfExists<std::vector<int>>("axes");
    if (first_node_attr_axes.empty()) return;
    auto second_node_attr_axes =
        second_op_node->Op()->GetAttrIfExists<std::vector<int>>("axes");
    if (second_node_attr_axes.empty()) return;
    second_op_node->Op()->RenameInput(first_out_var_node->Name(),
                                      first_in_var_node->Name());
    second_node_attr_axes.insert(second_node_attr_axes.begin(),
                                 first_node_attr_axes.begin(),
                                 first_node_attr_axes.end());
    second_op_node->Op()->SetAttr("axes", second_node_attr_axes);
    IR_NODE_LINK_TO(first_in_var_node, second_op_node);
    GraphSafeRemoveNodes(graph, {first_op_node, first_out_var_node});
    delete_counts++;
  };
  gpd(graph, handler);
  delete_op_count += delete_counts;
  if (delete_counts > 0) {
    LOG(INFO) << "--- delete " << delete_counts << " repeated "
              << "unsqueeze2"
              << " ops";
  }
}
void FusedContinuousSameOpsPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int repeat_time = 0;
  int total_delete_op_count = 0;
  // This pass needs to loop run until there are no nodes in the graph that need
  // to be deleted.
  while (true) {
    delete_op_count = 0;
    FusedReshapeOps(graph);
    FusedUnsqueezeOps(graph);
    LOG(INFO) << "Round " << repeat_time++
              << ": delete op counts: " << delete_op_count;
    total_delete_op_count += delete_op_count;
    if (delete_op_count == 0) {
      LOG(INFO) << "--- no nodes need to delete --- break";
      break;  // No node need to delete.
    }
  }
  LOG(INFO) << "Total delete op counts: " << total_delete_op_count;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fused_continuous_same_ops_pass,
              paddle::framework::ir::FusedContinuousSameOpsPass);

REGISTER_PASS_CAPABILITY(fused_continuous_same_ops_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "reshape2", 0))
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "unsqueeze2", 0));
