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
#include <iostream>
#include <queue>
#include <stack>
#include <unordered_set>

namespace common {

// depth-first search visitor
template <typename NodeType>
class DfsWalker final {
 public:
  DfsWalker(const DfsWalker&) = delete;
  DfsWalker(DfsWalker&&) = delete;

  using NodeHandlerType = std::function<void(NodeType)>;
  using NodesVisitorType =
      std::function<void(NodeType, const NodeHandlerType&)>;

  DfsWalker(const NodesVisitorType& VisitNextNodes)
      : VisitNextNodes_(VisitNextNodes) {}

  void operator()(NodeType node, const NodeHandlerType& NodeHandler) const {
    std::array<NodeType, 1> nodes{node};
    (*this)(nodes.begin(), nodes.end(), NodeHandler, [&](NodeType) {});
  }

  template <typename NodeIt>
  void operator()(NodeIt begin,
                  NodeIt end,
                  const NodeHandlerType& NodeHandler) const {
    (*this)(begin, end, NodeHandler, [&](NodeType) {});
  }

  // https://en.wikipedia.org/wiki/Depth-first_search
  template <typename NodeIt>
  void operator()(NodeIt begin,
                  NodeIt end,
                  const NodeHandlerType& NodeHandlerOnPush,
                  const NodeHandlerType& NodeHandlerOnPop) const {
    std::unordered_set<NodeType> discovered;
    struct Neighbours {
      NodeType producer;
      std::queue<NodeType> consumers;
    };
    std::stack<Neighbours> stack;
    const auto& TryPush = [&](NodeType node) {
      if (discovered.count(node) == 0) {
        discovered.insert(node);
        NodeHandlerOnPush(node);
        stack.push(Neighbours{.producer = node});
        VisitNextNodes_(node, [&](NodeType next_node) {
          stack.top().consumers.push(next_node);
        });
      }
    };
    for (NodeIt node_iter = begin; node_iter != end; ++node_iter) {
      TryPush(*node_iter);
      while (!stack.empty()) {
        auto* neighbours = &stack.top();
        if (neighbours->consumers.empty()) {
          NodeHandlerOnPop(neighbours->producer);
          stack.pop();
        } else {
          TryPush(neighbours->consumers.front());
          neighbours->consumers.pop();
        }
      }
    }
  }

 private:
  NodesVisitorType VisitNextNodes_;
};

}  // namespace common
