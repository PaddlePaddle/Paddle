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

#include "paddle/fluid/framework/ir/conv2d_fusion_cutlass_elementwise.h"
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void Conv2dFusionCutlassElementwiseFusePass::ApplyImpl(ir::Graph* graph) const {
  // This pass is used for cutlass, because cutlass can fuse conv + bias + silu
  bool cutlass_enable = Get<bool>("use_cutlass");
  if (!cutlass_enable) {
    return;
  }

  const std::string pattern_name = "conv_bias_act_elementwise";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  auto* conv_in = gpd.mutable_pattern()->NewNode("conv_in");

  std::string NAME = "conv2d_fusion";

  auto conv_filter = gpd.mutable_pattern()
                         ->NewNode("conv_filter")
                         ->assert_is_op_input(NAME, "Filter")
                         ->AsInput();
  auto conv_bias = gpd.mutable_pattern()
                       ->NewNode("conv_bias")
                       ->assert_is_op_input(NAME, "Bias")
                       ->AsInput();

  auto conv_op = gpd.mutable_pattern()->NewNode("conv_op")->assert_is_op(NAME);

  auto conv_out = gpd.mutable_pattern()
                      ->NewNode("conv_out")
                      ->assert_is_op_output(NAME)
                      ->assert_is_op_input("elementwise_add", "Y")
                      ->AsIntermediate();
  auto residual_input = gpd.mutable_pattern()
                            ->NewNode("residual_input")
                            ->assert_is_op_input("elementwise_add", "X");

  auto elementwise_add_op = gpd.mutable_pattern()
                                ->NewNode("elementwise_add_op")
                                ->assert_is_op("elementwise_add");

  auto elementwise_add_out = gpd.mutable_pattern()
                                 ->NewNode("elementwise_add_out")
                                 ->assert_is_op_output("elementwise_add")
                                 ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter, conv_bias}).LinksTo({conv_out});
  elementwise_add_op->LinksFrom({residual_input, conv_out})
      .LinksTo({elementwise_add_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    Node* conv_in_node = subgraph.at(conv_in);
    Node* conv_filter_node = subgraph.at(conv_filter);
    Node* conv_bias_node = subgraph.at(conv_bias);
    Node* conv_op_node = subgraph.at(conv_op);
    // Node* conv_out_node = subgraph.at(conv_out);
    Node* residual_input_node = subgraph.at(residual_input);
    Node* elementwise_add_op_node = subgraph.at(elementwise_add_op);
    Node* elementwise_add_out_node = subgraph.at(elementwise_add_out);

    OpDesc new_desc = *(conv_op_node->Op());
    // new_desc.SetType(NAME);
    // new_desc.SetInput("Input", {conv_in_node->Name()});
    // new_desc.SetInput("Filter", {conv_filter_node->Name()});
    // new_desc.SetInput("Bias", {conv_bias_node->Name()});
    // new_desc.SetAttr("activation", std::string("relu"));
    new_desc.SetInput("ResidualData", {residual_input_node->Name()});
    new_desc.SetOutput("Output", {elementwise_add_out_node->Name()});
    new_desc.Flush();

    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(conv_op_node);
    del_node_set.insert(elementwise_add_op_node);
    GraphSafeRemoveNodes(graph, del_node_set);

    auto fused_node = graph->CreateOpNode(&new_desc);
    IR_NODE_LINK_TO(conv_in_node, fused_node);
    IR_NODE_LINK_TO(conv_filter_node, fused_node);
    IR_NODE_LINK_TO(conv_bias_node, fused_node);
    IR_NODE_LINK_TO(residual_input_node, fused_node);
    IR_NODE_LINK_TO(fused_node, elementwise_add_out_node);
  };
  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv2d_fusion_cutlass_elementwise,
              paddle::framework::ir::Conv2dFusionCutlassElementwiseFusePass);
