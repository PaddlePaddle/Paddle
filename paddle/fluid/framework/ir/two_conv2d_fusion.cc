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

#include "paddle/fluid/framework/ir/two_conv2d_fusion.h"
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void TwoConv2dFusionFusePass::ApplyImpl(ir::Graph* graph) const {
  // This pass is used for cutlass, because cutlass can fuse conv + 1x1
  bool cutlass_enable = Get<bool>("use_cutlass");
  if (!cutlass_enable) {
    return;
  }

  const std::string pattern_name = "two_conv_fusion";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  std::string NAME = "conv2d_fusion_cutlass";
//  auto *scope = param_scope();

  auto* conv_in0 = gpd.mutable_pattern()->NewNode("conv_in0");
  auto conv_filter0 = gpd.mutable_pattern()
                         ->NewNode("conv_filter0")
                         ->assert_is_op_input(NAME, "Filter")
                         ->assert_is_persistable_var()->assert_more([&](Node *node) {
            return true;
            // auto filter_dims = scope->FindVar(node->Name())->GetMutable<phi::DenseTensor>()->dims();
            // return filter_dims[1] == 1 && filter_dims[2] == 1 && filter_dims[0] == 64;
  })->AsInput()->AsInput();


  auto conv_bias0 = gpd.mutable_pattern()
                       ->NewNode("conv_bias0")
                       ->assert_is_op_input(NAME, "Bias")
                       ->assert_is_persistable_var()
                       ->AsInput();
  auto conv_op0 = gpd.mutable_pattern()->NewNode("conv_op0")
                                       ->assert_is_op(NAME);

  auto conv_out0 = gpd.mutable_pattern()
                      ->NewNode("conv_out0")
                      ->assert_is_op_output(NAME)
                      ->assert_has_n_outputs(1)
                      ->AsIntermediate();

  auto conv_filter1 = gpd.mutable_pattern()
                         ->NewNode("conv_filter1")
                         ->assert_is_op_input(NAME, "Filter")
                         ->assert_is_persistable_var()
                         ->assert_more([&](Node *node) {
            
            return true;
          //  auto filter_dims = scope->FindVar(node->Name())->GetMutable<phi::DenseTensor>()->dims();
          //  return filter_dims[1] == 1 && filter_dims[2] == 1 && filter_dims[0] == 64;
  })->AsInput();

  auto conv_bias1 = gpd.mutable_pattern()
                       ->NewNode("conv_bias1")
                       ->assert_is_op_input(NAME, "Bias")
                       ->assert_is_persistable_var()
                       ->AsInput();

  auto conv_op1 = gpd.mutable_pattern()->NewNode("conv_op1")->assert_is_op(NAME)
                                      ->assert_has_n_inputs(3);
  
  auto conv_out1 = gpd.mutable_pattern()
                      ->NewNode("conv_out1")
                      ->assert_is_op_output(NAME)
                      ->AsOutput();

  conv_op0->LinksFrom({conv_in0, conv_filter0, conv_bias0}).LinksTo({conv_out0});
  conv_op1->LinksFrom({conv_out0, conv_filter1, conv_bias1})
      .LinksTo({conv_out1});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    Node* conv_in_node0 = subgraph.at(conv_in0);
    Node* conv_filter_node0 = subgraph.at(conv_filter0);
    Node* conv_bias_node0 = subgraph.at(conv_bias0);
    Node* conv_op_node0 = subgraph.at(conv_op0);
    Node* conv_out_node0 = subgraph.at(conv_out0);
    Node* conv_op_node1 = subgraph.at(conv_op1);
    Node* conv_filter_node1 = subgraph.at(conv_filter1);
    Node* conv_bias_node1 = subgraph.at(conv_bias1);
    Node* conv_out_node1 = subgraph.at(conv_out1);

    OpDesc new_desc = *(conv_op_node0->Op());
    new_desc.SetType("two_conv2d_fusion");
    new_desc.SetInput("Input", {conv_in_node0->Name()});
    new_desc.SetInput("Filter1", {conv_filter_node1->Name()});
    new_desc.SetInput("Bias1", {conv_bias_node1->Name()});
    new_desc.SetAttr("activation1", conv_op_node1->Op()->GetAttr("activation"));
    new_desc.SetAttr("strides1", conv_op_node1->Op()->GetAttr("strides"));
    new_desc.SetAttr("paddings1", conv_op_node1->Op()->GetAttr("paddings"));
    new_desc.SetOutput("Output", {conv_out_node1->Name()});
    new_desc.Flush();

    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(conv_op_node0);
    del_node_set.insert(conv_op_node1);
    del_node_set.insert(conv_out_node0);
    GraphSafeRemoveNodes(graph, del_node_set);

    auto fused_node = graph->CreateOpNode(&new_desc);
    IR_NODE_LINK_TO(conv_in_node0, fused_node);
    IR_NODE_LINK_TO(conv_filter_node0, fused_node);
    IR_NODE_LINK_TO(conv_bias_node0, fused_node);
    IR_NODE_LINK_TO(conv_filter_node1, fused_node);
    IR_NODE_LINK_TO(conv_bias_node1, fused_node);
    IR_NODE_LINK_TO(fused_node, conv_out_node1);
  };
  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(two_conv2d_fusion,
              paddle::framework::ir::TwoConv2dFusionFusePass);
