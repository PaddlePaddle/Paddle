// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/identity_scale_op_clean_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void IdentityScaleOpCleanPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("identity_scale_op_clean", graph);

  // pre_op -> scale_in -> scale_op -> scale_out
  // ->
  // pre_op -> scale_out
  GraphPatternDetector detector;
  auto scale_in =
      detector.mutable_pattern()
          ->NewNode("scale_in")
          ->assert_is_op_input("scale")
          ->assert_has_n_outputs(1)
          ->assert_more([](Node* x) {
            for (auto* op : x->inputs) {
              auto op_type = op->Op()->Type();
              if (op_type == "conditional_block" || op_type == "while") {
                return false;
              }
            }
            return true;
          });
  auto scale_op = detector.mutable_pattern()
                      ->NewNode("scale_fuse")
                      ->assert_is_op("scale")
                      ->assert_op_attr<float>("scale", 1.)
                      ->assert_op_attr<float>("bias", 0.);
  auto scale_out = detector.mutable_pattern()
                       ->NewNode("scale_out")
                       ->assert_is_op_output("scale");

  scale_op->LinksFrom({scale_in}).LinksTo({scale_out});

  int found_subgraph_count = 0;
  GraphPatternDetector::handle_t handler =
      [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
        Node* scale_op_var = subgraph.at(scale_op);
        Node* scale_in_var = subgraph.at(scale_in);
        Node* scale_out_var = subgraph.at(scale_out);
        const std::string scale_in_name = scale_in_var->Name();
        const std::string scale_out_name = scale_out_var->Name();
        // Remove links in graph
        GraphSafeRemoveNodes(graph, {scale_in_var, scale_op_var});
        // Modify pre_op_desc
        // Link pre_op directly to scale_out
        for (auto& node : graph->Nodes()) {
          if (node->IsOp()) {
            auto* op_desc = node->Op();
            auto out_vars_map = op_desc->Outputs();
            for (auto out_var_map : out_vars_map) {
              auto names = out_var_map.second;
              bool reset = false;
              for (size_t i = 0; i < names.size(); i++) {
                if (names[i] == scale_in_name) {
                  reset = true;
                  names[i] = scale_out_name;
                  break;
                }
              }
              if (reset) {
                op_desc->SetOutput(out_var_map.first, names);
                op_desc->Flush();
                IR_NODE_LINK_TO(node, scale_out_var);
                break;
              }
            }
          }
        }
        found_subgraph_count++;
      };

  detector(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(identity_scale_op_clean_pass,
              paddle::framework::ir::IdentityScaleOpCleanPass);
REGISTER_PASS_CAPABILITY(identity_scale_op_clean_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().EQ(
            "scale", 0));
