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
#include <utility>

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
  //                    last_op      input var
  std::vector<std::pair<ir::Node*, ir::Node*>> to_remove_unused_input;

  for (ir::Node* n : graph->Nodes()) {
    if (n->NodeType() == ir::Node::Type::kVariable) continue;
    for (ir::Node* in_var : n->inputs) {
      if (in_var->NodeType() == ir::Node::Type::kOperation) continue;
      for (ir::Node* upstream_op : in_var->inputs) {
        if (upstream_op->Name() == "assign") {
          VLOG(3) << "find path: " << upstream_op->Name() << "->"
                  << in_var->Name() << "->" << n->Name();
          // NOTE: This is a special case!!!
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
            // upstream_op->inputs.clear();
          } else {
            // connected by in/out, case like:
            // condition_op -> learning_rate -> assign -> final_learning_rate ->
            // sgd
            auto it =
                std::find(to_remove.begin(), to_remove.end(), upstream_op);
            if (it == to_remove.end()) {
              to_remove.push_back(upstream_op);
            }
            // connect origin var before assign to current op
            for (ir::Node* v : upstream_op->inputs) {
              VLOG(3) << "connect var: " << v->Name();
              auto upstream_it =
                  std::find(v->outputs.begin(), v->outputs.end(), upstream_op);
              if (upstream_it != v->outputs.end()) {
                v->outputs.erase(upstream_it);
              }
              v->outputs.push_back(n);
              n->inputs.push_back(v);
            }
            // // remove later out side the iterator
            to_remove_unused_input.push_back(std::make_pair(n, in_var));
          }
        }
      }
    }
  }

  for (ir::Node* n : to_remove) {
    n->inputs.clear();
    n->outputs.clear();
  }

  for (std::pair<ir::Node*, ir::Node*> np : to_remove_unused_input) {
    auto it =
        std::find(np.first->inputs.begin(), np.first->inputs.end(), np.second);
    if (it != np.first->inputs.end()) {
      np.first->inputs.erase(it);
    }
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
