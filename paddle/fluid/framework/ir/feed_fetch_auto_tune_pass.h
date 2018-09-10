#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * This Pass is used to automatically prune the graph and insert feed and fetch
 * ops. Currently, it is used in inference scenarios.
 */
class FeedFetchAutoTunePass final : public Pass {
 protected:
  ~FeedFetchAutoTunePass() = default;

  std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const override;

 private:
  bool Prune(Graph* graph, const std::vector<std::string>& feeds,
             const std::vector<std::string>& fetches);
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
      OpDesc desc;
      op->SetType("feed");
      op->SetInput("X", {item.first});
      op->SetOutput("Out", {var_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      auto* op = graph->CreateOpNode( OpDesc *op_desc )
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
