// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <queue>
#include "paddle/cinn/operator_fusion/pattern_graph.h"

namespace cinn::fusion {

struct NodePattern {};
struct EdgePattern {};
struct GraphPattern {};     // not implemented.
struct NodePairPattern {};  // not implemented.
struct ReverseTopoNodePairPattern {};

template <typename Kind, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm {};

template <typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePattern, GraphMatcher, GraphOperation> {
  PatternGraph* graph_;
  PatternNodePtrSet visited_nodes;

  explicit SearchAlgorithm(PatternGraph* graph) {
    VLOG(4) << "Create NodePattern algorithm.";
    graph_ = graph;
  }

  PatternNodePtr FindMatchedNode() {
    for (PatternNodePtr iter_node : graph_->all_pattern_nodes()) {
      if (GraphMatcher()(*graph_, iter_node) &&
          !visited_nodes.count(iter_node)) {
        visited_nodes.insert(iter_node);
        VLOG(4) << "Find Matched Node: " << iter_node->id() << "(" << iter_node
                << ")";
        return iter_node;
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return nullptr;
  }

  void operator()() {
    while (true) {
      PatternNodePtr node = FindMatchedNode();
      if (node == nullptr) {
        break;
      }
      GraphOperation()(graph_, node);
    }
  }
};

template <typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePairPattern, GraphMatcher, GraphOperation> {
  PatternGraph* graph_;
  std::set<std::pair<PatternNodePtr, PatternNodePtr>> visited_node_pair;
  explicit SearchAlgorithm(PatternGraph* graph) {
    VLOG(4) << "Create NodePairPattern algorithm.";
    graph_ = graph;
  }
  std::optional<std::pair<PatternNodePtr, PatternNodePtr>> FindMatchedPair() {
    for (PatternNodePtr i : graph_->all_pattern_nodes()) {
      for (PatternNodePtr j : graph_->all_pattern_nodes()) {
        if (i == j) continue;
        const auto& pair = std::make_pair(i, j);
        if (GraphMatcher()(*graph_, i, j) && !visited_node_pair.count(pair)) {
          visited_node_pair.insert(pair);
          VLOG(4) << "Find Matched Node Pair: (" << i->id() << ", " << j->id()
                  << ")";
          return pair;
        }
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return {};
  }
  void operator()() {
    while (true) {
      const auto& node = FindMatchedPair();
      if (!node.has_value()) break;
      const auto& [i, j] = node.value();
      GraphOperation()(graph_, i, j);
    }
  }
};

template <typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<ReverseTopoNodePairPattern,
                       GraphMatcher,
                       GraphOperation> {
  PatternGraph* graph_;
  std::queue<PatternNodePtr> reverse_topo_queue;
  std::unordered_map<PatternNodePtr, int> degree;

  explicit SearchAlgorithm(PatternGraph* graph) {
    VLOG(4) << "Create ReverseTopoNodePairPattern algorithm.";
    graph_ = graph;
    for (const auto& node : graph_->all_pattern_nodes()) {
      degree[node] = node->downstream().size();
      if (degree[node] == 0) {
        reverse_topo_queue.push(node);
      }
    }
  }

  void operator()() {
    while (!reverse_topo_queue.empty()) {
      PatternNodePtr node = reverse_topo_queue.front();
      reverse_topo_queue.pop();

      for (const auto& upstream : node->upstream()) {
        degree[upstream]--;
        if (degree[upstream] == 0) {
          reverse_topo_queue.push(upstream);
        }
      }

      auto fusion_candidates = node->downstream();
      for (const auto& downstream : fusion_candidates) {
        if (GraphMatcher()(*graph_, node, downstream)) {
          node = GraphOperation()(graph_, node, downstream);
        }
      }
    }
  }
};

template <typename Kind, typename GraphMatcher, typename GraphOperation>
void GraphTransformer(PatternGraph* graph) {
  VLOG(4) << "Start GraphTransformer...";
  auto algo = SearchAlgorithm<Kind, GraphMatcher, GraphOperation>(graph);
  algo();
}

}  // namespace cinn::fusion
