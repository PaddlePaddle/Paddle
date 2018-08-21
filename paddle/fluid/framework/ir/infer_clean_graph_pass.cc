// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class InferCleanGraphPass : public Pass {
 public:
  virtual ~InferCleanGraphPass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const {
    PADDLE_ENFORCE(graph.get());

    auto is_valid_node = [](Node* x) {
      return x && IsControlDepVar(*x) && x->IsVar() && !x->Var();
    };

    std::unordered_set<Node*> invalid_nodes;
    for (auto* node : graph->Nodes()) {
      if (is_valid_node(node)) {
        invalid_nodes.insert(node);
      }
    }

    // remove nodes from the graph.
    for (auto* node : invalid_nodes) {
      graph->RemoveNode(node);
    }

    // clean edges.
    for (auto* node : graph->Nodes()) {
      CleanEdges(&node->inputs, invalid_nodes);
      CleanEdges(&node->outputs, invalid_nodes);
    }

    return graph;
  }

  void CleanEdges(std::vector<Node*>* nodes,
                  const std::unordered_set<Node*>& to_remove) const {
    auto it = std::remove_if(nodes->begin(), nodes->end(),
                             [&](Node* x) { return to_remove.count(x); });
    nodes->erase(it, nodes->end());
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(infer_clean_graph_pass,
              paddle::framework::ir::InferCleanGraphPass);
