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

#include <queue>
#include "graph_viz_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

const static char kFeedHolderName[] = "feed";
const static char kFetchHolderName[] = "fetch";

const static char kFeedAttr[] = "__feed_names__";
const static char kFetchAttr[] = "__fetch_names__";
/*
 * This Pass is used to automatically prune the graph and insert feed and fetch
 * ops. Currently, it is used in inference scenarios.
 * NOTE It also suffers from the naming bug, that is the operator's output
 * variable have the same name of input, so the prune algorithm can't tell
 * whether to keep the input or output.
 * TODO(Superjomn) Fix that bug.
 */
class FeedFetchAutoPrunePass final : public Pass {
 protected:
  ~FeedFetchAutoPrunePass() = default;

  std::unique_ptr<Graph> ApplyImpl(
      std::unique_ptr<Graph> graph) const override {
    RemoveFeedFetchOps(graph.get());
    PADDLE_ENFORCE(!DetectFeedFetchOps(graph.get()));
    auto feeds = graph->Get<std::vector<std::string>>(kFeedAttr);
    auto fetches = graph->Get<std::vector<std::string>>(kFetchAttr);
    Prune(graph.get(), feeds, fetches);
    InsertFeedFetchOps(graph.get(), feeds, fetches);
    return graph;
  }

 private:
  // Remove the original feed fetch operators.
  bool RemoveFeedFetchOps(Graph *graph) const {
    std::unordered_set<const Node *> nodes_to_rm;
    for (auto &node : GraphTraits::DFS(*graph)) {
      if (node.IsOp() &&
          (node.Op()->Type() == "feed" || node.Op()->Type() == "fetch")) {
        nodes_to_rm.insert(&node);
      }
    }

    GraphSafeRemoveNodes(graph, nodes_to_rm);
    return true;
  }

  std::set<Node *> DFS(
      const std::vector<Node *> &starts, const Graph &graph,
      std::function<const std::vector<Node *> &(Node *)> &&get_nexts) const {
    LOG(INFO) << "DFS begin " << starts.size();
    std::queue<Node *> queue;
    std::set<Node *> res;
    for (auto *x : starts) {
      queue.push(x);
    }

    while (!queue.empty()) {
      auto *x = queue.front();
      queue.pop();
      res.insert(x);
      for (auto *out : get_nexts(x)) {
        if (!res.count(out)) {
          queue.push(out);
        }
      }
    }
    LOG(INFO) << "DFS end";
    return res;
  }

  std::set<Node *> ExtractNodesByName(
      const std::unordered_set<std::string> &names, const Graph &graph) const {
    std::set<Node *> res;
    for (auto *node : graph.Nodes()) {
      if (node->IsVar() && names.count(node->Name())) {
        res.insert(node);
      }
    }
    return res;
  }

  // Extract the nodes that starts from `feeds` and retches `fetches`.
  bool Prune(Graph *graph, const std::vector<std::string> &feeds,
             const std::vector<std::string> &fetches) const {
    auto nodes = ExtractDependedNodes(feeds, fetches, graph);

    // For debug only.
    auto &marked_nodes = GetMarkedNodes(graph);
    for (auto *node : nodes) {
      marked_nodes.insert(node);
    }

    std::unordered_set<const Node *> nodes_to_rm;
    // check all the nodes' inputs and outputs in this graph.
    for (auto *x : nodes) {
      if (!x->IsOp()) continue;
      for (auto *in : x->inputs) {
        if (!nodes.count(in)) {
          LOG(INFO) << in->Name() << " not contained";
          LOG(ERROR) << "the input of operator is not contained, failed to "
                        "deduce the dependency";
          return false;
        }
      }
    }
    // Remove unneeded nodes.
    for (auto *x : graph->Nodes()) {
      if (!nodes.count(x)) {
        nodes_to_rm.insert(x);
      }
    }
    GraphSafeRemoveNodes(graph, nodes_to_rm);

    return true;
  }

  std::set<Node *> ExtractDependedNodes(const std::vector<std::string> &feeds,
                                        const std::vector<std::string> &fetches,
                                        Graph *graph) const {
    // Extract all the nodes needed.
    auto start_points = ExtractNodesByName(
        std::unordered_set<std::string>(feeds.begin(), feeds.end()), *graph);
    auto end_points = ExtractNodesByName(
        std::unordered_set<std::string>(fetches.begin(), fetches.end()),
        *graph);

    auto forward_depends = DFS(
        std::vector<Node *>(start_points.begin(), start_points.end()), *graph,
        [](Node *x) -> const std::vector<Node *> & { return x->outputs; });

    auto backward_depends =
        DFS(std::vector<Node *>(end_points.begin(), end_points.end()), *graph,
            [](Node *x) -> const std::vector<Node *> & { return x->inputs; });

    std::vector<Node *> common_deps_vec;
    std::set_intersection(forward_depends.begin(), forward_depends.end(),
                          backward_depends.begin(), backward_depends.end(),
                          std::back_inserter(common_deps_vec));
    std::set<Node *> common_deps(common_deps_vec.begin(),
                                 common_deps_vec.end());
    // Check the start points and end points are within the common dependencies.
    for (auto *x : start_points) {
      if (!common_deps.count(x)) {
        LOG(FATAL) << string::Sprintf(
            "Wrong feeds for the target fetches, might need other inputs as "
            "feeds.");
      }
    }

    for (auto *x : end_points) {
      if (!common_deps.count(x)) {
        LOG(FATAL) << string::Sprintf(
            "Wrong feeds for the target fetches, might need other inputs as "
            "feeds.");
      }
    }

    // Include necessary parameters.
    for (auto *node : common_deps) {
      if (node->IsVar()) continue;
      for (auto *in : node->inputs) {
        if (in->Var()->Persistable()) {
          common_deps.insert(in);
        }
      }
    }

    return common_deps;
  }

  bool InsertFeedFetchOps(Graph *graph, const std::vector<std::string> &feeds,
                          const std::vector<std::string> &fetches) const {
    LOG(INFO) << "insert feed fetch";
    // Get all the Var Nodes that feeds or fetches.
    std::map<std::string, int> feed_index, fetch_index;
    for (size_t i = 0; i < feeds.size(); i++) {
      feed_index[feeds[i]] = i;
    }
    for (size_t i = 0; i < fetches.size(); i++) {
      fetch_index[fetches[i]] = i;
    }
    std::unordered_set<std::string> feed_set(feeds.begin(), feeds.end()),
        fetch_set(fetches.begin(), fetches.end());
    std::unordered_map<std::string, Node *> feed_map, fetch_map;

    for (auto *node : graph->Nodes()) {
      if (node->IsVar()) {
        if (feed_set.count(node->Name())) {
          feed_map[node->Name()] = node;
        } else if (fetch_set.count(node->Name())) {
          fetch_map[node->Name()] = node;
        }
      }
    }

    // Create feed and fetch ops.
    for (auto &item : feed_map) {
      OpDesc op_desc;
      op_desc.SetType("feed");
      op_desc.SetInput("X", {item.first});
      op_desc.SetOutput("Out", {item.second->Name()});
      op_desc.SetAttr("col", {feed_index.at(item.second->Name())});
      // op_desc.CheckAttrs();

      auto *op = graph->CreateOpNode(&op_desc);
      IR_NODE_LINK_TO(op, item.second);
    }

    for (auto &item : fetch_map) {
      OpDesc op_desc;
      op_desc.SetType("fetch");
      op_desc.SetInput("X", {item.second->Name()});
      op_desc.SetOutput("Out", {"fetch"});
      op_desc.SetAttr("col",
                      {static_cast<int>(fetch_index.at(item.second->Name()))});
      // op_desc.CheckAttrs();

      auto *op = graph->CreateOpNode(&op_desc);
      IR_NODE_LINK_TO(item.second, op);
    }
    return true;
  }

  bool DetectFeedFetchOps(Graph *graph) const {
    int feed_count{0};
    int fetch_count{0};
    for (auto *node : graph->Nodes()) {
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
