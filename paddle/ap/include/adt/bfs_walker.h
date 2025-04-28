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
#include "paddle/ap/include/adt/adt.h"

namespace ap::adt {

// breadth-first search visitor
template <typename NodeType>
class BfsWalker final {
 public:
  BfsWalker(const BfsWalker&) = delete;
  BfsWalker(BfsWalker&&) = delete;

  using NodeHandlerType = std::function<adt::Result<adt::Ok>(NodeType)>;
  using NodesVisitorType =
      std::function<adt::Result<adt::Ok>(NodeType, const NodeHandlerType&)>;

  BfsWalker(const NodesVisitorType& VisitNextNodesVal)
      : VisitNextNodes(VisitNextNodesVal) {}

  adt::Result<adt::Ok> operator()(NodeType node,
                                  const NodeHandlerType& NodeHandler) const {
    std::array<NodeType, 1> nodes{node};
    return (*this)(nodes.begin(), nodes.end(), NodeHandler);
  }

  template <typename NodeIt>
  adt::Result<adt::Ok> operator()(NodeIt begin,
                                  NodeIt end,
                                  const NodeHandlerType& NodeHandler) const {
    std::queue<NodeType> node_queue;
    std::unordered_set<NodeType> queued_nodes;
    const auto& TryEnqueueNode = [&](NodeType node) -> adt::Result<adt::Ok> {
      if (queued_nodes.count(node) == 0) {
        node_queue.push(node);
        queued_nodes.insert(node);
      }
      return adt::Ok{};
    };
    for (NodeIt iter = begin; iter != end; ++iter) {
      ADT_RETURN_IF_ERR(TryEnqueueNode(*iter));
    }
    while (!node_queue.empty()) {
      NodeType node = node_queue.front();
      node_queue.pop();
      ADT_RETURN_IF_ERR(NodeHandler(node));
      ADT_RETURN_IF_ERR(VisitNextNodes(node, TryEnqueueNode));
    }
    return adt::Ok{};
  }

  NodesVisitorType VisitNextNodes;
};

}  // namespace ap::adt
