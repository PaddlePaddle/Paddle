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

#include "paddle/fluid/framework/ir/split_gelu_mul_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#define GET_IR_NODE(node__) \
  GET_IR_NODE_FROM_SUBGRAPH(node__, node__, split_gelu_mul_pattern);
#define GET_NODES               \
  GET_IR_NODE(split);           \
  GET_IR_NODE(split_out0);      \
  GET_IR_NODE(split_out1);      \
  GET_IR_NODE(gelu);            \
  GET_IR_NODE(gelu_out);        \
  GET_IR_NODE(elementwise_mul); \
  GET_IR_NODE(elementwise_mul_out);

namespace paddle {
namespace framework {
namespace ir {
class Node;

namespace patterns {

struct SplitGeluMulPattern : public PatternBase {
  SplitGeluMulPattern(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "split_gelu") {}

  PDNode *operator()(PDNode *x);

  // declare operator node's name
  PATTERN_DECL_NODE(split);
  PATTERN_DECL_NODE(split_out0);
  PATTERN_DECL_NODE(split_out1);
  PATTERN_DECL_NODE(gelu);
  PATTERN_DECL_NODE(gelu_out);
  PATTERN_DECL_NODE(elementwise_mul);
  PATTERN_DECL_NODE(elementwise_mul_out);
};

PDNode *SplitGeluMulPattern::operator()(PDNode *x) {
  // Create nodes for split op.
  auto *split = pattern->NewNode(split_repr())->assert_is_op("split");
  auto *split_out0 = pattern->NewNode(split_out0_repr())
                         ->AsOutput()
                         ->assert_is_op_nth_output("split", "Out", 0);
  auto *split_out1 = pattern->NewNode(split_out1_repr())
                         ->AsOutput()
                         ->assert_is_op_nth_output("split", "Out", 1);
  // Add links for split op.
  split->LinksFrom({x}).LinksTo({split_out0, split_out1});

  // Create nodes for gelu op.
  split_out1->assert_is_op_input("gelu");
  auto *gelu = pattern->NewNode(gelu_repr())->assert_is_op("gelu");

  auto *gelu_out = pattern->NewNode(gelu_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("gelu", "Out");

  // Add links for gelu op.
  gelu->LinksFrom({split_out1}).LinksTo({gelu_out});

  // Create nodes for elementwise_mul op.
  gelu_out->assert_is_op_input("elementwise_mul");
  split_out0->assert_is_op_input("elementwise_mul");
  auto *elementwise_mul =
      pattern->NewNode(elementwise_mul_repr())->assert_is_op("elementwise_mul");

  auto *elementwise_mul_out =
      pattern->NewNode(elementwise_mul_out_repr())
          ->AsOutput()
          ->assert_is_op_output("elementwise_mul", "Out");

  // Add links for elementwise_mul op.
  elementwise_mul->LinksFrom({gelu_out, split_out0})
      .LinksTo({elementwise_mul_out});
  return elementwise_mul_out;
}

}  // namespace patterns

int SplitGeluMulFusePass::ApplyPattern(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      platform::errors::InvalidArgument(
          "The input graph of SplitGeluMulFusePass should not be "
          "nullptr."));
  GraphPatternDetector gpd;
  FusePassBase::Init(scope_name_, graph);
  PDNode *x = gpd.mutable_pattern()
                  ->NewNode("x")
                  ->assert_is_op_input("split", "X")
                  ->AsInput();
  patterns::SplitGeluMulPattern split_gelu_mul_pattern(gpd.mutable_pattern(),
                                                       scope_name_);
  split_gelu_mul_pattern(x);
  int fuse_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "split gelu mul in op compat failed.";
      return;
    }
    GET_NODES;

    std::unordered_set<const Node *> del_node_set = {
        split, split_out0, split_out1, gelu, gelu_out, elementwise_mul};

    OpDesc split_gelu_desc;
    split_gelu_desc.SetType("split_gelu");
    split_gelu_desc.SetInput("X", {subgraph.at(x)->Name()});
    split_gelu_desc.SetOutput("Out", {elementwise_mul_out->Name()});
    split_gelu_desc.Flush();

    auto split_gelu_node = graph->CreateOpNode(&split_gelu_desc);
    IR_NODE_LINK_TO(subgraph.at(x), split_gelu_node);
    IR_NODE_LINK_TO(split_gelu_node, elementwise_mul_out);
    GraphSafeRemoveNodes(graph, del_node_set);
    ++fuse_count;
  };
  gpd(graph, handler);
  return fuse_count;
}
void SplitGeluMulFusePass::ApplyImpl(ir::Graph *graph) const {
  int fuse_count = ApplyPattern(graph);
  AddStatis(fuse_count);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(split_gelu_mul_pass, paddle::framework::ir::SplitGeluMulFusePass);
REGISTER_PASS_CAPABILITY(split_gelu_mul_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("split", 0)
            .EQ("gelu", 0)
            .LE("elementwise_mul", 1));
