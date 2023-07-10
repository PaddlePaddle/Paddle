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

#include "paddle/fluid/framework/ir/identity_op_clean_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void IdentityOpCleanPass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("identity_scale_op_clean", graph);

  // pre_op -> useless_op_in -> useless_op -> useless_op_out
  // ->
  // pre_op -> useless_op_out
  GraphPatternDetector detector;
  auto useless_op_in =
      detector.mutable_pattern()
          ->NewNode("useless_op_in")
          ->assert_has_n_outputs(1)
          ->assert_var_not_persistable()
          ->assert_more([](Node* x) {
            for (auto* op : x->inputs) {
              auto op_type = op->Op()->Type();
              if (op_type == "conditional_block" || op_type == "while") {
                return false;
              }
            }
            return true;
          });

  // This useless_op must have only one input and one output!
  auto useless_op =
      detector.mutable_pattern()
          ->NewNode("useless_op")
          ->assert_has_n_inputs(1)
          ->assert_has_n_outputs(1)
          ->assert_more([](Node* x) {
            if (!x->IsOp()) {
              return false;
            }
            if (x->Op()->Type() == "scale") {
              auto scale = x->Op()->GetAttrIfExists<float>("scale");
              auto bias = x->Op()->GetAttrIfExists<float>("bias");
              if (bias == 0 && scale == 1) {
                return true;
              }
            }
            if (x->Op()->Type() == "cast") {
              auto in_dtype = x->Op()->GetAttrIfExists<int>("in_dtype");
              auto out_dtype = x->Op()->GetAttrIfExists<int>("out_dtype");
              if (in_dtype == out_dtype) {
                return true;
              }
            }
            if (x->Op()->Type() == "c_identity") {
              return true;
            }
            // you can add more cases here.
            return false;
          });
  auto useless_op_out = detector.mutable_pattern()->NewNode("useless_op_out");

  useless_op->LinksFrom({useless_op_in}).LinksTo({useless_op_out});

  int found_subgraph_count = 0;
  GraphPatternDetector::handle_t handler =
      [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
        Node* useless_op_var = subgraph.at(useless_op);
        Node* useless_op_in_var = subgraph.at(useless_op_in);
        Node* useless_op_out_var = subgraph.at(useless_op_out);
        const std::string useless_op_in_name = useless_op_in_var->Name();
        const std::string useless_op_out_name = useless_op_out_var->Name();

        if (useless_op_in_var->inputs.size() != 1L) return;
        auto pre_op_node = useless_op_in_var->inputs[0];

        // Link pre_op directly to scale_out
        auto* pre_op_desc = pre_op_node->Op();
        IR_NODE_LINK_TO(pre_op_node, useless_op_out_var);
        // Modify pre_op_desc
        pre_op_desc->RenameOutput(useless_op_in_var->Name(),
                                  useless_op_out_var->Name());

        // Remove nodes in graph
        GraphSafeRemoveNodes(graph, {useless_op_in_var, useless_op_var});

        found_subgraph_count++;
      };

  detector(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(identity_op_clean_pass,
              paddle::framework::ir::IdentityOpCleanPass);
REGISTER_PASS_CAPABILITY(identity_op_clean_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("scale", 0)
            .LE("c_identity", 1));
