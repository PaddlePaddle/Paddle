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
#include "paddle/cinn/operator_fusion/pattern_graph.h"

namespace cinn::fusion {

struct NodePattern {};
struct EdgePattern {};
struct GraphPattern {};     // not implemented.
struct NodePairPattern {};  // not implemented.
struct ReverseTopoNodePairPattern {};

template <typename Kind,
          typename Phrase,
          typename GraphMatcher,
          typename GraphOperation>
struct SearchAlgorithm {};

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePattern, Phrase, GraphMatcher, GraphOperation> {
  PatternGraph<Phrase>* graph_;
  PatternNodePtrSet<Phrase> visited_nodes;

  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create NodePattern algorithm.";
    graph_ = graph;
  }

  PatternNodePtr<Phrase> FindMatchedNode() {
    for (PatternNodePtr<Phrase> iter_node : graph_->all_pattern_nodes()) {
      if (GraphMatcher()(*graph_, iter_node) &&
          !visited_nodes.count(iter_node)) {
        visited_nodes.insert(iter_node);
        VLOG(4) << "Find Matched Node: " << iter_node;
        return iter_node;
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return nullptr;
  }

  void operator()() {
    while (true) {
      PatternNodePtr<Phrase> node = FindMatchedNode();
      if (node == nullptr) {
        break;
      }
      GraphOperation()(graph_, node);
    }
  }
};

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePairPattern, Phrase, GraphMatcher, GraphOperation> {
  PatternGraph<Phrase>* graph_;
  std::set<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
      visited_node_pair;
  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create NodePairPattern algorithm.";
    graph_ = graph;
  }
  std::optional<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
  FindMatchedPair() {
    for (PatternNodePtr<Phrase> i : graph_->all_pattern_nodes()) {
      for (PatternNodePtr<Phrase> j : graph_->all_pattern_nodes()) {
        if (i == j) continue;
        const auto& pair = std::make_pair(i, j);
        if (GraphMatcher()(*graph_, i, j) && !visited_node_pair.count(pair)) {
          visited_node_pair.insert(pair);
          VLOG(4) << "Find Matched Node Pair: (" << i << ", " << j << ")";
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

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<ReverseTopoNodePairPattern,
                       Phrase,
                       GraphMatcher,
                       GraphOperation> {
  PatternGraph<Phrase>* graph_;
  std::queue<PatternNodePtr<Phrase>> reverse_topo_nodes;

  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create ReverseTopoNodePairPattern algorithm.";
    graph_ = graph;

    // Do reverse topological sort, and store the results in reverse_topo_nodes.
    std::unordered_map<PatternNodePtr<Phrase>, int>
        unvisited_nodes_to_out_degree;
    for (const auto& node_ptr : graph->all_pattern_nodes()) {
      unvisited_nodes_to_out_degree[node_ptr] = node_ptr->downstream().size();
    }

    while (!unvisited_nodes_to_out_degree.empty()) {
      const auto& it =
          std::find_if(unvisited_nodes_to_out_degree.begin(),
                       unvisited_nodes_to_out_degree.end(),
                       [&](const auto& pair) { return pair.second == 0; });
      reverse_topo_nodes.push(it->first);
      for (const auto& upstream : it->first->upstream()) {
        --unvisited_nodes_to_out_degree[upstream];
      }
      unvisited_nodes_to_out_degree.erase(it);
    }
  }

  std::optional<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
  FindMatchedPair() {
    //  Keep picking the front element of reverse_topo_nodes as candidate of
    //  upstream node. Please make sure that the downstream node is merged into
    //  the upstream node during merging, and the upstream node will not
    //  disappear after merging, else the logic here should be modified.
    while (!reverse_topo_nodes.empty()) {
      const auto& upstream_candidate = reverse_topo_nodes.front();

      // If the node has downstream, try searching for its candidate downstream
      // using GraphMatcher.
      if (!upstream_candidate->downstream().empty()) {
        for (const auto& downstream_candidate :
             upstream_candidate->downstream()) {
          if (GraphMatcher()(
                  *graph_, upstream_candidate, downstream_candidate)) {
            VLOG(4) << "Find Matched Node Pair: (" << upstream_candidate << ", "
                    << downstream_candidate << ")";
            return std::make_pair(upstream_candidate, downstream_candidate);
          }
        }
      }
      reverse_topo_nodes.pop();
    }

    VLOG(4) << "Can't find matched node any more.";
    return {};
  }

  void operator()() {
    while (true) {
      const auto& node = FindMatchedPair();
      if (!node.has_value()) break;
      const auto& [upstream, downstream] = node.value();
      GraphOperation()(graph_, upstream, downstream);
    }
  }
};

template <typename Kind,
          typename Phrase,
          typename GraphMatcher,
          typename GraphOperation>
void GraphTransformer(PatternGraph<Phrase>* graph) {
  VLOG(4) << "Start GraphTransformer...";
  auto alog =
      SearchAlgorithm<Kind, Phrase, GraphMatcher, GraphOperation>(graph);
  alog();
}

}  // namespace cinn::fusion
