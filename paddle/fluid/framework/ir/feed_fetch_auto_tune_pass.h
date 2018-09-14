#include <queue>
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

const static char kFeedHolderName[] = "feed";
const static char kFetchHolderName[] = "fetch";

/*
 * This Pass is used to automatically prune the graph and insert feed and fetch
 * ops. Currently, it is used in inference scenarios.
 */
class FeedFetchAutoTunePass final : public Pass {
 protected:
  ~FeedFetchAutoTunePass() = default;

  std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const override;

 private:
  // Remove the original feed fetch operators.
  bool RemoveFeedFetchOps(Graph* graph) {
    std::vector<Node*> nodes_to_rm;
    for (auto node : GraphTraits::DFS(*graph)) {
      if (node.IsOp() &&
          (node.Op()->Type() == "feed" || node.Op()->Type() == "fetch")) {
        nodes_to_rm.push_back(&node);
      }
    }

    for (auto* node : nodes_to_rm) {
      // TODO(Superjomn) check the edges are included.
      graph->RemoveNode(node);
    }
  }

  std::set<Node*> DFS(const std::vector<Node*>& starts, const Graph& graph,
                      std::function<std::vector<Node*>(Node*)>&& get_nexts) {
    std::queue<Node*> queue;
    std::unordered_set<Node*> set;
    std::set<Node*> res;
    for (auto* x : starts) {
      queue.push(x);
    }

    while (!queue.empty()) {
      auto* x = queue.front();
      res.insert(x);
      for (auto* out : get_nexts(x)) {
        if (!set.count(out)) {
          queue.push(out);
        }
      }
    }
    return res;
  }

  std::set<Node*> ExtractNodesByName(
      const std::unordered_set<std::string>& names, const Graph& graph) {
    std::set<Node*> res;
    for (auto* node : graph.Nodes()) {
      if (node->IsVar() && names.count(node->Name())) {
        res.insert(node);
      }
    }
    return res;
  }

  // Extract the nodes that starts from `feeds` and retches `fetches`.
  bool Prune(Graph* graph, const std::vector<std::string>& feeds,
             const std::vector<std::string>& fetches) {
    auto nodes = ExtractDependedNodes(feeds, fetches, graph);
    std::vector<Node*> nodes_to_rm;
    // Remove unneeded nodes.
    for (auto* x : graph->Nodes()) {
      if (!nodes.count(x)) {
        nodes_to_rm.push_back(x);
      }
    }
    for (auto* x : nodes_to_rm) {
      graph->RemoveNode(x);
    }
    // check all the nodes' inputs and outputs in this graph.
    for (auto* x : nodes) {
      if (!x->IsOp()) continue;
      for (auto* in : x->inputs) {
        if (!nodes.count(in)) {
          LOG(ERROR) << "the input of operator is not contained, failed to "
                        "deduce the dependency";
          return false;
        }
      }
    }

    return true;
  }

  std::set<Node*> ExtractDependedNodes(const std::vector<std::string>& feeds,
                                       const std::vector<std::string>& fetches,
                                       Graph* graph) {
    // Extract all the nodes needed.
    auto start_points = ExtractNodesByName(
        std::unordered_set<std::string>(feeds.begin(), feeds.end()), *graph);
    auto end_points = ExtractNodesByName(
        std::unordered_set<std::string>(fetches.begin(), fetches.end()),
        *graph);

    auto forward_depends =
        DFS(std::vector<Node*>(start_points.begin(), start_points.end()),
            *graph, [](Node* x) { return x->outputs; });

    auto backward_depends =
        DFS(std::vector<Node*>(start_points.begin(), start_points.end()),
            *graph, [](Node* x) { return x->inputs; });

    std::set<Node*> common_deps;
    std::set_intersection(forward_depends.begin(), forward_depends.end(),
                          backward_depends.begin(), backward_depends.end(),
                          std::back_inserter(common_deps));
    // Check the start points and end points are within the common dependencies.
    bool all_in = false;
    for (auto* x : start_points) {
      if (!common_deps.count(x)) {
        all_in = false;
        break;
      }
    }
    if (!all_in) {
      LOG(FATAL) << string::Sprintf(
          "Wrong feeds for the target fetches, might need other inputs as "
          "feeds.");
    }

    for (auto* x : end_points) {
      if (!common_deps.count(x)) {
        LOG(FATAL) << string::Sprintf(
            "Wrong feeds for the target fetches, might need other inputs as "
            "feeds.");
      }
    }

    return common_deps;
  }

  bool InsertFeedFetchOps(Graph* graph, const std::vector<std::string>& feeds,
                          const std::vector<std::string>& fetches) {
    // Get all the Var Nodes that feeds or fetches.
    std::unordered_set<std::string> feed_set(feeds.begin(), feeds.end()),
        fetch_set(fetches.begin(), fetches.end());
    std::unordered_map<std::string, Node *> feed_map, fetch_map;
    for (auto* node : graph->Nodes()) {
      if (node->IsVar()) {
        if (feed_set.count(node->Name())) {
          feed_map[node->Name()] = node;
        } else if (fetch_set.count(node->Name())) {
          fetch_map[node->Name()] = node;
        }
      }
    }

    // Create feed and fetch ops.
    for (auto& item : feed_map) {
      OpDesc op_desc;
      op_desc.SetType("feed");
      op_desc.SetInput("X", {item.first});
      op_desc.SetOutput("Out", {item.second->Name()});
      op_desc.SetAttr("col", {static_cast<int>(i)});
      op_desc.CheckAttrs();

      auto* op = graph->CreateOpNode(&op_desc);
    }
  }

  bool DetectFeedFetchOps(Graph* graph) {
    int feed_count{0};
    int fetch_count{0};
    for (auto* node : graph->Nodes()) {
      if (node->IsOp() && node->Op()) {
        if (node->Op()->Type() == "feed") {
          ++feed_count;
        } else if (node->Op()->Type() == "fetch") {
          ++fetch_count;
        }
      }
    }
    return feed_count > 0 && fetch_count > 0;
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
