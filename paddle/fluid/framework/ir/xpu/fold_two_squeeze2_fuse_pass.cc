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

#include "paddle/fluid/framework/ir/xpu/fold_two_squeeze2_fuse_pass.h"
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

struct TwoSqueeze2FusePattern : public PatternBase {
  TwoSqueeze2FusePattern(PDPattern* pattern, const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(squeeze2_1);
  PATTERN_DECL_NODE(squeeze2_2);
  // declare variable node's name
  PATTERN_DECL_NODE(x);
  PATTERN_DECL_NODE(squeeze2_1_out);
  PATTERN_DECL_NODE(squeeze2_2_out);
};

TwoSqueeze2FusePattern::TwoSqueeze2FusePattern(PDPattern* pattern,
                                               const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  auto* x = pattern->NewNode(x_repr())
                ->assert_is_op_input("squeeze2", "X")
                ->AsInput()
                ->assert_more([](Node* node) {
                  auto squeeze2_in_x_shape = node->Var()->GetShape();
                  size_t squeeze2_in_rank = squeeze2_in_x_shape.size();
                  bool nice_shape = squeeze2_in_x_shape[1] == 1 &&
                                    squeeze2_in_x_shape[2] == 74 &&
                                    squeeze2_in_x_shape[3] == 1;
                  return squeeze2_in_rank == 4 && nice_shape;
                });
  auto* squeeze2_1 = pattern->NewNode(squeeze2_1_repr())
                         ->assert_is_op("squeeze2")
                         ->assert_more([](Node* node) {
                           auto* op_desc = node->Op();
                           return op_desc->GetAttrIfExists<std::vector<int>>(
                                      "axes") == std::vector<int>{3};
                         });
  auto* squeeze2_1_out = pattern->NewNode(squeeze2_1_out_repr())
                             ->assert_is_op_output("squeeze2", "Out")
                             ->assert_has_n_outputs(1)
                             ->assert_is_op_input("squeeze2", "X");
  squeeze2_1->LinksFrom({x}).LinksTo({squeeze2_1_out});
  auto* squeeze2_2 = pattern->NewNode(squeeze2_2_repr())
                         ->assert_is_op("squeeze2")
                         ->assert_more([](Node* node) {
                           auto* op_desc = node->Op();
                           return op_desc->GetAttrIfExists<std::vector<int>>(
                                      "axes") == std::vector<int>{1};
                         });
  auto* squeeze2_2_out = pattern->NewNode(squeeze2_2_out_repr())
                             ->assert_is_op_output("squeeze2", "Out")
                             ->assert_has_n_outputs(1);
  squeeze2_2->LinksFrom({squeeze2_1_out}).LinksTo({squeeze2_2_out});
}

}  // namespace patterns

void FoldTwoSqueeze2FusePass::FoldTwoSqueeze2(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::TwoSqueeze2FusePattern pattern(gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FoldTwoSqueeze2FusePass";
    // declare operator node's name
    GET_IR_NODE(squeeze2_1);
    GET_IR_NODE(squeeze2_2);
    // declare variable node's name
    GET_IR_NODE(x);
    GET_IR_NODE(squeeze2_1_out);
    GET_IR_NODE(squeeze2_2_out);

    auto* block = squeeze2_1->Op()->Block();
    // Generate reshape2 op
    framework::OpDesc reshape2_op_desc(block);
    reshape2_op_desc.SetType("reshape2");
    reshape2_op_desc.SetInput("X", {x->Name()});
    reshape2_op_desc.SetAttr("shape", std::vector<int>{-1, 74});
    reshape2_op_desc.SetOutput("Out", {squeeze2_2_out->Name()});

    auto* reshape2 = graph->CreateOpNode(&reshape2_op_desc);

    IR_NODE_LINK_TO(x, reshape2);
    IR_NODE_LINK_TO(reshape2, squeeze2_2_out);
    // delete useless node
    std::unordered_set<const Node*> delete_nodes = {
        squeeze2_1, squeeze2_2, squeeze2_1_out};
    GraphSafeRemoveNodes(graph, delete_nodes);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

void FoldTwoSqueeze2FusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);

  FoldTwoSqueeze2(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fold_two_squeeze2_fuse_pass,
              paddle::framework::ir::FoldTwoSqueeze2FusePass);

REGISTER_PASS_CAPABILITY(fold_two_squeeze2_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "squeeze2", 0));
