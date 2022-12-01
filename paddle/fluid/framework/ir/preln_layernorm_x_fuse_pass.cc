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

#include "paddle/fluid/framework/ir/preln_layernorm_x_fuse_pass.h"

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

struct PrelnLayerNormX : public PatternBase {
  PrelnLayerNormX(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "preln_layernorm_x") {}

  void operator()(PDNode *x, PDNode *y, const std::string &norm_type);
  // declare operator node's name
  PATTERN_DECL_NODE(elementwise_bias);
  PATTERN_DECL_NODE(elementwise0);
  PATTERN_DECL_NODE(elementwise1);
  PATTERN_DECL_NODE(layer_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(elementwise0_out);
  PATTERN_DECL_NODE(elementwise1_out);

  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
};

void PrelnLayerNormX::operator()(PDNode *x,
                                 PDNode *y,
                                 const std::string &norm_type) {
  auto *elementwise1 =
      pattern->NewNode(elementwise1_repr())->assert_is_op("elementwise_add");
  auto *elementwise1_out_var =
      pattern->NewNode(elementwise1_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_is_op_input(norm_type, "X");

  elementwise1->LinksFrom({x, y}).LinksTo({elementwise1_out_var});
  // Create nodes for layer_norm op.
  auto *layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op(norm_type);
  auto *layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input(norm_type, "Bias");

  auto *layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input(norm_type, "Scale");

  auto *layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output(norm_type, "Y");

  // Add links for layer_norm op.
  layer_norm
      ->LinksFrom(
          {elementwise1_out_var, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo({layer_norm_out_var});
}

}  // namespace patterns

int PrelnLayerNormXFusePass::ApplyLayerNormShiftPattern(
    ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("preln_layernorm_x_fuse", graph);

  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  PDNode *x = nullptr;
  PDNode *y = nullptr;

  x = gpd.mutable_pattern()
          ->NewNode("preln_layernorm_x_fuse/x")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "X");

  y = gpd.mutable_pattern()
          ->NewNode("preln_layernorm_x_fuse/y")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "Y");
  patterns::PrelnLayerNormX fused_pattern(gpd.mutable_pattern(),
                                          "preln_layernorm_x_fuse");
  fused_pattern(x, y, "layernorm_shift_partition");

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    VLOG(4) << "handle preln layernorm x fuse";

    GET_IR_NODE_FROM_SUBGRAPH(elementwise1, elementwise1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise1_out, elementwise1_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_pattern);

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "preln_layernorm_x_fuse pass in op compat failed.";
      return;
    }

    std::unordered_set<const Node *> del_node_set;
    // Create an PrelnLayerNormX op node
    OpDesc new_desc(*layer_norm->Op());
    new_desc.SetType("preln_layernorm_shift_partition");
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetOutput("Out_0", {elementwise1_out->Name()});
    new_desc.SetOutput("Out_1", {layer_norm_out->Name()});
    new_desc.RemoveOutput("Y");
    new_desc.Flush();

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(elementwise1);
    del_node_set.insert(layer_norm);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, layer_norm_out);
    IR_NODE_LINK_TO(fused_node, elementwise1_out);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

int PrelnLayerNormXFusePass::ApplyMergeLayerNormPattern(
    ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("preln_layernorm_x_fuse", graph);

  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  PDNode *x = nullptr;
  PDNode *y = nullptr;

  x = gpd.mutable_pattern()
          ->NewNode("preln_layernorm_x_fuse/x")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "X");

  y = gpd.mutable_pattern()
          ->NewNode("preln_layernorm_x_fuse/y")
          ->AsInput()
          ->assert_var_not_persistable()
          ->assert_is_op_input("elementwise_add", "Y");
  patterns::PrelnLayerNormX fused_pattern(gpd.mutable_pattern(),
                                          "preln_layernorm_x_fuse");
  fused_pattern(x, y, "merge_layernorm");

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    VLOG(4) << "handle preln layernorm x fuse";

    GET_IR_NODE_FROM_SUBGRAPH(elementwise1, elementwise1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise1_out, elementwise1_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_pattern);

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "preln_layernorm_x_fuse pass in op compat failed.";
      return;
    }
    std::unordered_set<const Node *> del_node_set;
    // Create an PrelnLayerNormX op node
    OpDesc new_desc(*layer_norm->Op());
    new_desc.SetType("skip_merge_layernorm");
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetOutput("Out", {layer_norm_out->Name()});
    new_desc.RemoveOutput("Y");
    new_desc.Flush();

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(elementwise1);
    del_node_set.insert(layer_norm);
    del_node_set.insert(elementwise1_out);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, layer_norm_out);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void PrelnLayerNormXFusePass::ApplyImpl(ir::Graph *graph) const {
  FusePassBase::Init("preln_layernorm_x_fuse", graph);
  int found_subgraph_count = ApplyLayerNormShiftPattern(graph);
  found_subgraph_count += ApplyMergeLayerNormPattern(graph);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(preln_layernorm_x_fuse_pass,
              paddle::framework::ir::PrelnLayerNormXFusePass);
REGISTER_PASS_CAPABILITY(preln_layernorm_x_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "elementwise_add", 1));
