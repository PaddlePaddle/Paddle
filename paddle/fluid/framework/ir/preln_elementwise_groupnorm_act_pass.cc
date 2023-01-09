/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/preln_elementwise_groupnorm_act_pass.h"
#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct PrelnGroupNormAct : public PatternBase {
  PrelnGroupNormAct(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "preln_groupnorm_act") {}

  void operator()(PDNode *x, PDNode *y);
  // declare operator node's name
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(group_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(elementwise_out);

  PATTERN_DECL_NODE(group_norm_bias);
  PATTERN_DECL_NODE(group_norm_scale);
  PATTERN_DECL_NODE(group_norm_out);
  PATTERN_DECL_NODE(act);
  PATTERN_DECL_NODE(act_out);
};

void PrelnGroupNormAct::operator()(PDNode *x, PDNode *y) {
  auto *elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_add");

  auto *elementwise_out_var =
      pattern->NewNode(elementwise_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_is_op_input("group_norm", "X");

  elementwise->LinksFrom({x, y}).LinksTo({elementwise_out_var});
  // Create nodes for group_norm op.
  auto *group_norm =
      pattern->NewNode(group_norm_repr())->assert_is_op("group_norm");
  auto *group_norm_bias_var = pattern->NewNode(group_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("group_norm", "Bias");

  auto *group_norm_scale_var = pattern->NewNode(group_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("group_norm", "Scale");

  auto *group_norm_out_var = pattern->NewNode(group_norm_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output("group_norm", "Y")
                                 ->assert_is_op_input("silu", "X");

  // Add links for group_norm op.
  group_norm
      ->LinksFrom(
          {elementwise_out_var, group_norm_bias_var, group_norm_scale_var})
      .LinksTo({group_norm_out_var});

  auto *act = pattern->NewNode(act_repr())->assert_is_op("silu");
  auto *act_out = pattern->NewNode(act_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("silu", "Out");

  act->LinksFrom({group_norm_out_var}).LinksTo({act_out});
}

}  // namespace patterns

int PrelnGroupNormActFusePass::ApplyGNSiluPattern(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("preln_groupnorm_silu_fuse", graph);

  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  PDNode *x = nullptr;
  PDNode *y = nullptr;

  x = gpd.mutable_pattern()
          ->NewNode("preln_groupnorm_act_fuse/x")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "X");

  y = gpd.mutable_pattern()
          ->NewNode("preln_groupnorm_act_fuse/y")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "Y");

  patterns::PrelnGroupNormAct fused_pattern(gpd.mutable_pattern(),
                                            "preln_groupnorm_act_fuse");
  fused_pattern(x, y);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    VLOG(4) << "handle preln groupnorm act fuse";

    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(group_norm, group_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(group_norm_bias, group_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        group_norm_scale, group_norm_scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(group_norm_out, group_norm_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, fused_pattern);

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "preln groupnorm act pass in op compat failed.";
      return;
    }

    std::unordered_set<const Node *> del_node_set;
    // Create an preln_groupnorm_act op node
    OpDesc new_desc(*group_norm->Op());
    new_desc.SetType("preln_groupnorm_act");
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetOutput("Out_0", {elementwise_out->Name()});
    new_desc.SetOutput("Out_1", {act_out->Name()});
    new_desc.RemoveOutput("Y");
    new_desc.Flush();

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(elementwise);
    del_node_set.insert(group_norm);
    del_node_set.insert(group_norm_out);
    del_node_set.insert(act);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    IR_NODE_LINK_TO(group_norm_scale, fused_node);
    IR_NODE_LINK_TO(group_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, act_out);
    IR_NODE_LINK_TO(fused_node, elementwise_out);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void PrelnGroupNormActFusePass::ApplyImpl(ir::Graph *graph) const {
  FusePassBase::Init("preln_groupnorm_act_fuse_pass", graph);
  int found_subgraph_count = ApplyGNSiluPattern(graph);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(preln_elementwise_groupnorm_act_pass,
              paddle::framework::ir::PrelnGroupNormActFusePass);
REGISTER_PASS_CAPABILITY(preln_elementwise_groupnorm_act_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("silu", 0)
            .EQ("group_norm", 0));
