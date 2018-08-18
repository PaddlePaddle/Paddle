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

    auto is_ctrol_dep_node = [](Node* x) {
      return x && x->IsVar() && !x->Var();
    };

    std::unordered_set<Node*> ctrol_dep_nodes;
    for (auto* node : graph->Nodes()) {
      if (is_ctrol_dep_node(node)) {
        ctrol_dep_nodes.insert(node);
      }
    }

    // remove nodes from the graph.
    for (auto* node : ctrol_dep_nodes) {
      graph->RemoveNode(node);
    }

    // clean edges.
    for (auto* node : graph->Nodes()) {
      CleanEdges(&node->inputs, ctrol_dep_nodes);
      CleanEdges(&node->outputs, ctrol_dep_nodes);
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
