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

#include "paddle/fluid/framework/ir/preln_residual_bias_fuse_pass.h"

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

struct PrelnResidualBias : public PatternBase {
  PrelnResidualBias(PDPattern *pattern,
                    const std::string &name_scope,
                    bool with_bias)
      : PatternBase(pattern, name_scope, "preln_residual_bias") {
    with_bias_ = with_bias;
  }

  void operator()(PDNode *x, PDNode *y);

  bool with_bias_;
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
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
};

void PrelnResidualBias::operator()(PDNode *x, PDNode *y) {
  PDNode *elementwise0 = nullptr;
  PDNode *elementwise_bias_var = nullptr;
  PDNode *elementwise0_out_var = nullptr;
  // Create nodes for elementwise add op.
  if (with_bias_) {
    elementwise0 =
        pattern->NewNode(elementwise0_repr())->assert_is_op("elementwise_add");
    elementwise_bias_var = pattern->NewNode(elementwise_bias_repr())
                               ->assert_is_op_input("elementwise_add", "Y")
                               ->assert_is_persistable_var();
    elementwise0_out_var = pattern->NewNode(elementwise0_out_repr())
                               ->assert_is_op_output("elementwise_add")
                               ->assert_is_op_input("elementwise_add")
                               ->assert_more([](Node *x) {
                                 if (x->outputs.size() == 1) {
                                   return true;
                                 } else {
                                   return false;
                                 }
                               });
  } else {
    elementwise0_out_var = y;
  }

  auto *elementwise1 =
      pattern->NewNode(elementwise1_repr())->assert_is_op("elementwise_add");
  auto *elementwise1_out_var = pattern->NewNode(elementwise1_out_repr())
                                   ->assert_is_op_input("layer_norm", "X");
  // Add links for elementwise_add op.
  if (with_bias_) {
    elementwise0->LinksFrom({y, elementwise_bias_var})
        .LinksTo({elementwise0_out_var});
    elementwise1_out_var->assert_is_op_output("elementwise_add");
  }

  elementwise1->LinksFrom({x, elementwise0_out_var})
      .LinksTo({elementwise1_out_var});
  // Create nodes for layer_norm op.
  auto *layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto *layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto *layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");

  auto *layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output("layer_norm", "Y");
  auto *layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto *layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");
  // Add links for layer_norm op.
  layer_norm
      ->LinksFrom(
          {elementwise1_out_var, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo(
          {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});
}

}  // namespace patterns

int PrelnResidualBiasFusePass::ApplyPattern(ir::Graph *graph,
                                            bool with_bias) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("preln_residual_bias_fuse", graph);

  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  PDNode *x = nullptr;
  PDNode *y = nullptr;
  if (with_bias) {
    x = gpd.mutable_pattern()
            ->NewNode("preln_residual_bias_fuse/x")
            ->AsInput()
            ->assert_is_op_input("elementwise_add")
            ->assert_var_not_persistable();
    y = gpd.mutable_pattern()
            ->NewNode("preln_residual_bias_fuse/y")
            ->AsInput()
            ->assert_is_op_input("elementwise_add", "X")
            ->assert_var_not_persistable();
  } else {
    x = gpd.mutable_pattern()
            ->NewNode("preln_residual_bias_fuse/x")
            ->AsInput()
            ->assert_is_op_input("elementwise_add", "X");

    y = gpd.mutable_pattern()
            ->NewNode("preln_residual_bias_fuse/y")
            ->AsInput()
            ->assert_is_op_input("elementwise_add", "Y");
  }
  patterns::PrelnResidualBias fused_pattern(
      gpd.mutable_pattern(), "preln_residual_bias_fuse", with_bias);
  fused_pattern(x, y);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    VLOG(4) << "handle PrelnResidualBias fuse";
    Node *elementwise_bias = nullptr;
    Node *elementwise0 = nullptr;
    Node *elementwise0_out = nullptr;
    if (with_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(
          tmp_elementwise_bias, elementwise_bias, fused_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(tmp_elementwise0, elementwise0, fused_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(
          tmp_elementwise0_out, elementwise0_out, fused_pattern);
      elementwise_bias = tmp_elementwise_bias;
      elementwise0 = tmp_elementwise0;
      elementwise0_out = tmp_elementwise0_out;
    }
    GET_IR_NODE_FROM_SUBGRAPH(elementwise1, elementwise1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise1_out, elementwise1_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_scale, layer_norm_scale, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        layer_norm_variance, layer_norm_variance, fused_pattern);

    // We can not accept that two or more layer_norm is connected to
    // elementwise1_out. This will lead to two or more PrelnResidualBias
    // patterns is found near elementwise1_out, and these patterns will interact
    // on each other, so we make below check to ensure only one
    // PrelnResidualBias pattern is delalted with.
    for (auto op : elementwise1_out->inputs) {
      if (op->Name() == "preln_residual_bias") return;
    }

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "preln_residual_bias pass in op compat failed.";
      return;
    }

    std::unordered_set<const Node *> del_node_set;
    // Create an PrelnResidualBias op node
    OpDesc new_desc;
    new_desc.SetType("preln_residual_bias");
    // inputs
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetInput("Scale", {layer_norm_scale->Name()});
    new_desc.SetInput("Bias", {layer_norm_bias->Name()});
    if (with_bias) {
      new_desc.SetInput("EleBias", {elementwise_bias->Name()});
    }
    // outputs
    new_desc.SetOutput("Out_0", {layer_norm_out->Name()});
    new_desc.SetOutput("Out_1", {elementwise1_out->Name()});
    // attrs
    new_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));
    new_desc.SetAttr("begin_norm_axis",
                     layer_norm->Op()->GetAttr("begin_norm_axis"));
    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.
    if (with_bias) {
      del_node_set.insert(elementwise0);
      del_node_set.insert(elementwise0_out);
    }
    del_node_set.insert(elementwise1);
    del_node_set.insert(layer_norm);
    del_node_set.insert(layer_norm_mean);
    del_node_set.insert(layer_norm_variance);
    GraphSafeRemoveNodes(graph, del_node_set);
    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    if (with_bias) {
      IR_NODE_LINK_TO(elementwise_bias, fused_node);
    }
    IR_NODE_LINK_TO(layer_norm_scale, fused_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, layer_norm_out);
    IR_NODE_LINK_TO(fused_node, elementwise1_out);
    found_subgraph_count++;
  };

  gpd(graph, handler);
  return found_subgraph_count;
}

void PrelnResidualBiasFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("preln_residual_bias_fuse", graph);

  int found_subgraph_count = 0;
  found_subgraph_count = ApplyPattern(graph, true);
  found_subgraph_count += ApplyPattern(graph, false);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(preln_residual_bias_fuse_pass,
              paddle::framework::ir::PrelnResidualBiasFusePass);
REGISTER_PASS_CAPABILITY(preln_residual_bias_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("layer_norm", 0));
