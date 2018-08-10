#include "paddle/fluid/framework/ir/graph_pattern_detecter.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

PDNode* PDPattern::NewNode() {
  nodes_.emplace_back();
  auto* cur = &nodes_.back();
  cur->id = nodes_.size() - 1;
  return cur;
}

void PDPattern::AddEdge(PDNode* a, PDNode* b) {
  PADDLE_ENFORCE(a);
  PADDLE_ENFORCE(b);
  PADDLE_ENFORCE(a != b, "can't link between the same nodes.");
  edges_.emplace_back(a, b);
}

void GraphPatternDetecter::operator()(GraphPatternDetecter::handle_t handler) {}

bool GraphPatternDetecter::MarkPDNodesInGraph(const ir::Graph& graph) {
  if (graph.Nodes().empty()) return false;

  for (auto* node : TopologySortOperations(graph)) {
    for (const auto& pdnode : pattern_.nodes()) {
      if (pdnode.teller(node)) {
        pdnodes2nodes_[&pdnode].insert(node);
      }
    }
  }
  return !pdnodes2nodes_.empty();
}

struct HitGroup {
  std::unordered_map<Node*, PDNode*> roles;

  bool Match(Node* node, PDNode* pat) {
    return !roles.count(node) || roles.at(node) == pat;
  }

  void Register(Node* node, PDNode* pat) { roles[node] = pat; }
};

// Tell whether Node a links to b.
bool IsNodesLink(Node* a, Node* b) {
  for (auto* node : a->outputs) {
    if (b == node) return true;
  }
  return false;
}

std::vector<GraphPatternDetecter::subgraph_t>
GraphPatternDetecter::DetectPatterns() {
  // Init empty subgraphs.
  std::vector<GraphPatternDetecter::subgraph_t> result;
  std::vector<HitGroup> init_groups;
  auto* first_pnode = pattern_.edges().front().first;
  if (!pdnodes2nodes_.count(first_pnode)) {
    return result;
  }
  for (auto* node : pdnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.roles[node] = first_pnode;
    init_groups.emplace_back(group);
  }

  int step = 0;
  std::array<std::vector<HitGroup>, 2> bi_records;
  bi_records[0] = std::move(init_groups);

  for (const auto& edge : pattern_.edges()) {
    // Each role has two PDNodes, which indicates two roles.
    // Detect two Nodes that can match these two roles and they are connected.

    auto& cur_groups = bi_records[step % 2];
    auto& pre_groups = bi_records[1 - (step++ % 2)];

    cur_groups.clear();
    // source -> target
    for (Node* source : pdnodes2nodes_[edge.first]) {
      for (Node* target : pdnodes2nodes_[edge.second]) {
        // TODO(Superjomn) add some prune strategies.
        for (auto& group : pre_groups) {
          if (group.Match(source, edge.first) &&
              group.Match(target, edge.second) && IsNodesLink(source, target)) {
            HitGroup new_group;
            new_group = group;
            new_group.Register(source, edge.first);
            new_group.Register(target, edge.second);
            cur_groups.push_back(new_group);
            // TODO(Superjomn) need to unique
          }
        }
      }
    }
  }

  for (auto& group : bi_records[step % 2]) {
    GraphPatternDetecter::subgraph_t subgraph;
    for (auto& role : group.roles) {
      subgraph.emplace_back(role);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
