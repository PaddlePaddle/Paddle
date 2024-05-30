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

#include "paddle/fluid/framework/ir/silu_fuse_pass.h"
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir {

void SiluFusePass::ApplyImpl(ir::Graph* graph) const {
  // This pass is used for cutlass, because cutlass can fuse conv + bias + silu
  bool cutlass_enable = Get<bool>("use_cutlass");
  bool use_custom_device = Get<bool>("use_custom_device");
  if (!cutlass_enable && !use_custom_device) {
    return;
  }

  const std::string pattern_name = "silu_fuse";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  auto* sigmoid_in = gpd.mutable_pattern()->NewNode("sigmoid_in");
  auto sigmoid_op =
      gpd.mutable_pattern()->NewNode("sigmoid_op")->assert_is_op("sigmoid");
  auto sigmoid_out = gpd.mutable_pattern()
                         ->NewNode("sigmoid_out")
                         ->assert_is_op_output("sigmoid")
                         ->AsIntermediate();
  auto elementwise_mul_op = gpd.mutable_pattern()
                                ->NewNode("elementwise_mul_op")
                                ->assert_is_op("elementwise_mul");

  auto elementwise_mul_out = gpd.mutable_pattern()
                                 ->NewNode("elementwise_mul_out")
                                 ->assert_is_op_output("elementwise_mul")
                                 ->AsOutput();

  sigmoid_op->LinksFrom({sigmoid_in}).LinksTo({sigmoid_out});
  elementwise_mul_op->LinksFrom({sigmoid_in, sigmoid_out})
      .LinksTo({elementwise_mul_out});

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    Node* sigmoid_in_node = subgraph.at(sigmoid_in);
    Node* sigmoid_op_node = subgraph.at(sigmoid_op);
    Node* elementwise_mul_op_node = subgraph.at(elementwise_mul_op);
    Node* elementwise_mul_out_node = subgraph.at(elementwise_mul_out);

    OpDesc new_desc;
    new_desc.SetType("swish");
    new_desc.SetAttr("beta", 1.f);
    new_desc.SetInput("X", {sigmoid_in_node->Name()});
    new_desc.SetOutput("Out", {elementwise_mul_out_node->Name()});
    new_desc.Flush();

    std::unordered_set<const Node*> del_node_set;
    del_node_set.insert(sigmoid_op_node);
    del_node_set.insert(elementwise_mul_op_node);
    GraphSafeRemoveNodes(graph, del_node_set);

    auto fused_node = graph->CreateOpNode(&new_desc);
    IR_NODE_LINK_TO(sigmoid_in_node, fused_node);
    IR_NODE_LINK_TO(fused_node, elementwise_mul_out_node);
  };
  gpd(graph, handler);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(silu_fuse_pass, paddle::framework::ir::SiluFusePass);
