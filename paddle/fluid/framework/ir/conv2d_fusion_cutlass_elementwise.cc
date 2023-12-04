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
#include "paddle/fluid/framework/ir/cutlass_teller.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

//     conv_in
//       |
//  conv2d_fusion  residual_input
//             |        |
//         conv_out     |
//             |        |
//            elementwise_op
//                 |
//         elementwise_op_out

//
// -> fused to
//
//         conv_in     residual_input
//             |        |
//            conv2d_fusion
//                 |
//         elementwise_op_out

void Conv2dFusionCutlassElementwiseFusePass::ApplyImpl(ir::Graph* graph) const {
  // This pass is used for cutlass, because cutlass can fuse conv + bias + act0
  // + elementwise_op + act1
  bool cutlass_enable = Get<bool>("use_cutlass");
  if (!cutlass_enable) {
    return;
  }

  const std::string pattern_name = "conv2d_fusion_cutlass_elementwise";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  auto* conv_in = gpd.mutable_pattern()->NewNode("conv_in");

  std::string target_op = "conv2d_fusion";
  // std::unordered_set<std::string> cutlass_ele_set(
  //     {"elementwise_add"});

  auto conv_filter = gpd.mutable_pattern()
                         ->NewNode("conv_filter")
                         ->assert_is_op_input(target_op, "Filter")
                         ->AsInput();
  auto conv_bias = gpd.mutable_pattern()
                       ->NewNode("conv_bias")
                       ->assert_is_op_input(target_op, "Bias")
                       ->AsInput();

  auto conv_op =
      gpd.mutable_pattern()->NewNode("conv_op")->assert_is_op(target_op);

  auto conv_out = gpd.mutable_pattern()
                      ->NewNode("conv_out")
                      ->assert_is_op_output(target_op)
                      ->assert_is_op_input("elementwise_add", "Y")
                      ->AsIntermediate();
  auto residual_input = gpd.mutable_pattern()
                            ->NewNode("residual_input")
                            ->assert_is_op_input("elementwise_add", "X");

  auto elementwise_op = gpd.mutable_pattern()
                            ->NewNode("elementwise_op")
                            ->assert_is_op("elementwise_add");

  auto elementwise_op_out = gpd.mutable_pattern()
                                ->NewNode("elementwise_op_out")
                                ->assert_is_op_output("elementwise_add")
                                ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter, conv_bias}).LinksTo({conv_out});
  elementwise_op->LinksFrom({residual_input, conv_out})
      .LinksTo({elementwise_op_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    Node* conv_in_node = subgraph.at(conv_in);
    Node* conv_filter_node = subgraph.at(conv_filter);
    Node* conv_bias_node = subgraph.at(conv_bias);
    Node* conv_op_node = subgraph.at(conv_op);
    // Node* conv_out_node = subgraph.at(conv_out);
    Node* residual_input_node = subgraph.at(residual_input);
    Node* elementwise_op_node = subgraph.at(elementwise_op);
    Node* elementwise_op_out_node = subgraph.at(elementwise_op_out);

    auto* scope = param_scope();
    bool cutlass_can_fuse =
        CutlassTeller::Instance()->CbaeleCanSupport(conv_op_node->Op(),
                                                    scope,
                                                    elementwise_op_node->Name(),
                                                    "identity",
                                                    Get<int>("gpu_device_id"));

    if (!cutlass_can_fuse) {
      return;
    }

    OpDesc new_desc = *(conv_op_node->Op());
    auto activation = new_desc.GetAttrIfExists<std::string>("activation");
    new_desc.SetAttr("activation",
                     activation + std::string("_elementwise_add_identity"));
    new_desc.SetInput("ResidualData", {residual_input_node->Name()});
    new_desc.SetOutput("Output", {elementwise_op_out_node->Name()});
    new_desc.Flush();

    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(conv_op_node);
    del_node_set.insert(elementwise_op_node);
    GraphSafeRemoveNodes(graph, del_node_set);

    auto fused_node = graph->CreateOpNode(&new_desc);
    IR_NODE_LINK_TO(conv_in_node, fused_node);
    IR_NODE_LINK_TO(conv_filter_node, fused_node);
    IR_NODE_LINK_TO(conv_bias_node, fused_node);
    IR_NODE_LINK_TO(residual_input_node, fused_node);
    IR_NODE_LINK_TO(fused_node, elementwise_op_out_node);
  };
  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv2d_fusion_cutlass_elementwise,
              paddle::framework::ir::Conv2dFusionCutlassElementwiseFusePass);
