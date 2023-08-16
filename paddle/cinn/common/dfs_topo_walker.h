// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <array>
#include <functional>
#include <stack>
#include <unordered_set>

namespace cinn {
namespace common {

// DFS Topological order walker.
// Try to walk in a depth first manner while ensuring topological order.
// For example:
//   Graph:
//     0 -> 1
//     2 -> 3
//     0 -> 3
//     1 -> 3
//     3 -> 4
//   Start nodes: 0, 2
//   Walking order: 0 -> 1 -> 2 -> 3 -> 4
template <typename NodeType,
          typename NodeHash = std::hash<NodeType>,
          typename NodeEqual = std::equal_to<NodeType>>
class DfsTopoWalker final {
 public:
  DfsTopoWalker(const DfsTopoWalker&) = delete;
  DfsTopoWalker(DfsTopoWalker&&) = delete;

  using NodeHandlerType = std::function<void(NodeType)>;
  using NodesVisitorType =
      std::function<void(NodeType, const NodeHandlerType&)>;

  DfsTopoWalker(const NodesVisitorType& VisitPreNodes,
                const NodesVisitorType& VisitNextNodes)
      : VisitPreNodes_(VisitPreNodes), VisitNextNodes_(VisitNextNodes) {}

  // Start walking from 1 node and make every effort to access all nodes that
  // meet the walking rules.
  // If there are more than 1 nodes with a degree of 0 in a graph,
  // only one part will be accessed.
  // If you want to access the entire graph,
  // you need to provide all starting nodes.
  void operator()(NodeType node, const NodeHandlerType& NodeHandler) const {
    std::array<NodeType, 1> nodes{node};
    (*this)(nodes.begin(), nodes.end(), NodeHandler);
  }

  // Start walking from a collection of node and make every effort to access all
  // nodes that meet the walking rules.
  // If there are other start nodes in a graph,
  // some nodes on the graph will not be accessed.
  // If you want to access the entire graph,
  // you need to provide all starting nodes.
  template <typename NodeIt>
  void operator()(NodeIt begin,
                  NodeIt end,
                  const NodeHandlerType& NodeHandler) const {
    std::stack<NodeType> node_stack;
    std::unordered_set<NodeType, NodeHash, NodeEqual> visited;
    std::unordered_map<NodeType, int, NodeHash, NodeEqual> in_degree;
    const auto& InitInDegree = [&](NodeType node) {
      if (in_degree.count(node) == 0) {
        in_degree[node] = 0;
        VisitPreNodes_(node, [&](NodeType in_node) { ++in_degree[node]; });
      }
    };
    const auto& UpdateInDegree = [&](NodeType node) {
      InitInDegree(node);
      --in_degree[node];
    };
    const auto& TryPush = [&](NodeType node) {
      InitInDegree(node);
      if (visited.count(node) == 0 && in_degree[node] == 0) {
        node_stack.push(node);
        visited.insert(node);
      }
    };

    for (NodeIt iter = begin; iter != end; ++iter) {
      TryPush(*iter);
      while (!node_stack.empty()) {
        NodeType cur = node_stack.top();
        node_stack.pop();
        NodeHandler(cur);
        VisitNextNodes_(cur, UpdateInDegree);
        VisitNextNodes_(cur, TryPush);
      }
    }
  }

 private:
  NodesVisitorType VisitNextNodes_;
  NodesVisitorType VisitPreNodes_;
};

}  // namespace common
}  // namespace cinn
