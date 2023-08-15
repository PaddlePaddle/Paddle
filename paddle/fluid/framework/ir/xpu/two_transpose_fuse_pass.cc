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

#include "paddle/fluid/framework/ir/xpu/two_transpose_fuse_pass.h"
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

struct TwoTransposeFusePattern : public PatternBase {
  TwoTransposeFusePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(transpose_1);
  PATTERN_DECL_NODE(transpose_2);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(transpose_1_out);
  PATTERN_DECL_NODE(transpose_2_out);
};

TwoTransposeFusePattern::TwoTransposeFusePattern(PDPattern* pattern,
                                                 const std::string& name_scope)
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

void TwoTransposeFusePass::TwoTranspose(ir::Graph* graph) const {
  GraphPatternDetector gpd;

  patterns::TwoTransposeFusePattern pattern(gpd.mutable_pattern(), name_scope_);

  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle TwoTransposeFusePass";
    // declare operator node's name
    GET_IR_NODE(transpose_1);
    GET_IR_NODE(transpose_2);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(transpose_1_out);
    GET_IR_NODE(transpose_2_out);

    auto* block = transpose_1->Op()->Block();
    // Generate reshape2 op
    framework::OpDesc transpose_op_desc(block);
    transpose_op_desc.SetType("transpose2");
    transpose_op_desc.SetInput("X", {x->Name()});

    auto axis1 = transpose_1->Op()->GetAttrIfExists<std::vector<int>>("axis");
    auto axis2 = transpose_2->Op()->GetAttrIfExists<std::vector<int>>("axis");

    for (int i = 0; i < axis2.size(); i++) {
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
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void TwoTransposeFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  TwoTranspose(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(two_transpose_fuse_pass,
              paddle::framework::ir::TwoTransposeFusePass);

REGISTER_PASS_CAPABILITY(two_transpose_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "transpose2", 0));
