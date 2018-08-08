/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdio.h>
#include <algorithm>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/shrink_assign_depvar_pass.h"

namespace paddle {
namespace framework {
namespace ir {

std::vector<ir::Node*> GetUpstreamOps(const ir::Node* op) {
  std::vector<ir::Node*> upstreams;
  for (ir::Node* var : op->inputs) {
    for (ir::Node* upstream : var->inputs) {
      PADDLE_ENFORCE(upstream->NodeType() == ir::Node::Type::kOperation);
      upstreams.push_back(upstream);
    }
  }
  return upstreams;
}

std::unique_ptr<ir::Graph> ShrinkAssignDepvarPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  std::vector<ir::Node*> to_remove;

  for (ir::Node* n : graph->Nodes()) {
    if (n->NodeType() == ir::Node::Type::kVariable) continue;

    for (ir::Node* in_var : n->inputs) {
      for (ir::Node* upstream_op : in_var->inputs) {
        PADDLE_ENFORCE(upstream_op->NodeType() == ir::Node::Type::kOperation);
        if (upstream_op->Name() == "assign") {
          printf("find path: %s -> %s -> %s\n", upstream_op->Name().c_str(),
                 in_var->Name().c_str(), n->Name().c_str());
          // connected by dep_var, cases like:
          // condition_op -> learning_rate
          //              \-> assign -> final_learning_rate ->
          //                        \-> dep_var -> contition_op
          // then, use dep_var to connect ops around assign
          if (IsControlDepVar(*in_var)) {
            to_remove.push_back(upstream_op);
            // remove assign -> dep_var connect
            auto it = std::find(in_var->inputs.begin(), in_var->inputs.end(),
                                upstream_op);
            if (it != in_var->inputs.end()) {
              in_var->inputs.erase(it);
            }
            // add grand -> dep_var connect
            auto before_assign_ops = GetUpstreamOps(upstream_op);
            for (ir::Node* grand : before_assign_ops) {
              grand->outputs.push_back(in_var);
              in_var->inputs.push_back(grand);
            }
            // assign then can remove all inputs
            upstream_op->inputs.clear();
          } else {
            // connected by in/out, case like:
            // condition_op -> learning_rate -> assign -> final_learning_rate ->
            // sgd
            // can remove assign and final_learning_rate
            // TODO(typhoonzero): Can remove assign for performance
            //                    but need to change the operators input var
            //                    names.
          }
        }
      }
    }
  }

  for (ir::Node* n : to_remove) {
    n->outputs.clear();
  }

  for (ir::Node* n : to_remove) {
    for (ir::Node* out : n->outputs) {
      graph->RemoveNode(out);
    }
    graph->RemoveNode(n);
  }

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(shrink_assign_depvar_pass,
              paddle::framework::ir::ShrinkAssignDepvarPass);
