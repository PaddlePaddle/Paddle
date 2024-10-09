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

#include "paddle/fluid/framework/ir/xpu/duplicated_transpose_fuse_pass.h"
#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

namespace patterns {

struct DuplicatedTransposeFusePattern : public PatternBase {
  DuplicatedTransposeFusePattern(PDPattern* pattern,
                                 const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(transpose_1);
  PATTERN_DECL_NODE(transpose_2);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(transpose_1_out);
  PATTERN_DECL_NODE(transpose_2_out);
};

DuplicatedTransposeFusePattern::DuplicatedTransposeFusePattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())->assert_is_op_input("transpose2", "X");
  auto* transpose_1 =
      pattern->NewNode(transpose_1_repr())->assert_is_op("transpose2");
  auto* transpose_1_out = pattern->NewNode(transpose_1_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_has_n_outputs(1)
                              ->assert_is_op_input("transpose2", "X");
  transpose_1->LinksFrom({x}).LinksTo({transpose_1_out});
  auto* transpose_2 =
      pattern->NewNode(transpose_2_repr())->assert_is_op("transpose2");
  auto* transpose_2_out = pattern->NewNode(transpose_2_out_repr())
                              ->assert_is_op_output("transpose2", "Out")
                              ->assert_has_n_outputs(1);
  transpose_2->LinksFrom({transpose_1_out}).LinksTo({transpose_2_out});
}

}  // namespace patterns

void DuplicatedTransposeFusePass::DuplicatedTranspose(ir::Graph* graph) const {
  GraphPatternDetector gpd;

  patterns::DuplicatedTransposeFusePattern pattern(gpd.mutable_pattern(),
                                                   name_scope_);

  // int found_subgraph_count = 0;
  int delete_counts = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle DuplicatedTransposeFusePass";
    // declare operator node's name
    GET_IR_NODE(transpose_1);
    GET_IR_NODE(transpose_2);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(transpose_1_out);
    GET_IR_NODE(transpose_2_out);

    auto* block = transpose_1->Op()->Block();
    // Generate transpose2 op
    framework::OpDesc transpose_op_desc(block);
    transpose_op_desc.SetType("transpose2");
    transpose_op_desc.SetInput("X", {x->Name()});
    auto axis1 = transpose_1->Op()->GetAttrIfExists<std::vector<int>>("axis");
    if (axis1.empty()) return;
    auto axis2 = transpose_2->Op()->GetAttrIfExists<std::vector<int>>("axis");
    if (axis2.empty()) return;
    for (size_t i = 0; i < axis2.size(); i++) {
      axis2[i] = axis1[axis2[i]];
    }
    transpose_op_desc.SetAttr("axis", axis2);
    transpose_op_desc.SetOutput("Out", {transpose_2_out->Name()});

    auto* transpose = graph->CreateOpNode(&transpose_op_desc);

    IR_NODE_LINK_TO(x, transpose);
    IR_NODE_LINK_TO(transpose, transpose_2_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {
        transpose_1, transpose_2, transpose_1_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    delete_counts++;
  };
  gpd(graph, handler);
  delete_op_count += delete_counts;

  if (delete_counts > 0) {
    LOG(INFO) << "--- delete " << delete_counts << " repeated "
              << "transpose2"
              << " ops";
  }
}

void DuplicatedTransposeFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  int repeat_time = 0;
  int total_delete_op_count = 0;
  // This pass needs to loop run until there are no nodes in the graph that need
  // to be deleted.
  while (true) {
    delete_op_count = 0;
    DuplicatedTranspose(graph);
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

REGISTER_PASS(duplicated_transpose_fuse_pass,
              paddle::framework::ir::DuplicatedTransposeFusePass);

REGISTER_PASS_CAPABILITY(duplicated_transpose_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "transpose2", 0));
