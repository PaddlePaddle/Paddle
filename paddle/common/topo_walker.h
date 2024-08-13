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
#include <queue>
#include <unordered_set>

namespace common {

// Topological order visitor
template <typename NodeType>
class TopoWalker final {
 public:
  TopoWalker(const TopoWalker&) = default;
  TopoWalker(TopoWalker&&) = default;

  using NodeHandlerType = std::function<void(NodeType)>;
  using NodesVisitorType =
      std::function<void(NodeType, const NodeHandlerType&)>;

  TopoWalker(const NodesVisitorType& VisitPrevNodesValue,
             const NodesVisitorType& VisitNextNodesValue)
      : VisitPrevNodes(VisitPrevNodesValue),
        VisitNextNodes(VisitNextNodesValue) {}

  void operator()(NodeType node, const NodeHandlerType& NodeHandler) const {
    std::array<NodeType, 1> nodes{node};
    (*this)(nodes.begin(), nodes.end(), NodeHandler);
  }

  template <typename NodeIt>
  void operator()(NodeIt begin,
                  NodeIt end,
                  const NodeHandlerType& NodeHandler) const {
    std::queue<NodeType> node_queue;
    std::unordered_set<NodeType> queued_nodes;
    const auto& TryEnqueueNode = [&](NodeType node) {
      if (queued_nodes.count(node) == 0) {
        node_queue.push(node);
        queued_nodes.insert(node);
      }
    };
    for (NodeIt iter = begin; iter != end; ++iter) {
      TryEnqueueNode(*iter);
    }
    while (!node_queue.empty()) {
      NodeType node = node_queue.front();
      node_queue.pop();
      NodeHandler(node);
      VisitNextNodes(node, [&](NodeType node) {
        size_t num_unfinished_inputs = 0;
        VisitPrevNodes(node, [&](NodeType in_node) {
          num_unfinished_inputs += (queued_nodes.count(in_node) > 0 ? 0 : 1);
        });
        if (num_unfinished_inputs == 0) {
          TryEnqueueNode(node);
        }
      });
    }
  }

  NodesVisitorType VisitPrevNodes;
  NodesVisitorType VisitNextNodes;
};

}  // namespace common
