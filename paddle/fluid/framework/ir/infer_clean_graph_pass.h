#pragma once

#include <algorithm>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

class InferCleanGraphPass : public FusePassBase {
 public:
  virtual ~InferCleanGraphPass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const {
    FusePassBase::Init("original_graph", graph.get());
    PADDLE_ENFORCE(graph.get());

    auto is_valid_node = [](Node* x) {
      return x && IsControlDepVar(*x) && x->IsVar() && !x->Var();
    };

    std::unordered_set<const Node*> invalid_nodes;
    int valid_op = 0;
    for (auto* node : graph->Nodes()) {
      if (is_valid_node(node)) {
        invalid_nodes.insert(node);
      } else if (node->IsOp()) {
        // Collect all the operators to help tracking number of operators.
        ++valid_op;
      }
    }

    GraphSafeRemoveNodes(graph.get(), invalid_nodes);

    AddStatis(valid_op);

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
