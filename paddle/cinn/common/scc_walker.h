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

#include <glog/logging.h>

#include <functional>
#include <list>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/cinn/common/dfs_walker.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

// strong connected components visitor
template <typename NodeType>
class SccWalker final {
 public:
  SccWalker(const SccWalker&) = delete;
  SccWalker(SccWalker&&) = delete;

  using NodeHandlerType = std::function<void(NodeType)>;
  using NodesVisitorType =
      std::function<void(NodeType, const NodeHandlerType&)>;

  SccWalker(const NodesVisitorType& VisitPrevNodes,
            const NodesVisitorType& VisitNextNodes)
      : VisitPrevNodes_(VisitPrevNodes), VisitNextNodes_(VisitNextNodes) {}

  using SccHandlerType = std::function<void(const std::vector<NodeType>&)>;

  // https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm
  template <typename NodeIt>
  void operator()(NodeIt begin,
                  NodeIt end,
                  const SccHandlerType& SccHandler) const {
    const std::list<NodeType>& dfs_ordered_nodes = [&]() {
      std::list<NodeType> dfs_ordered_nodes;
      DfsVisitor<NodeType> visitor(VisitNextNodes_);
      visitor(
          begin,
          end,
          /*on push*/ [](NodeType) {},
          /*on pop*/
          [&](NodeType node) { dfs_ordered_nodes.push_front(node); });
      return dfs_ordered_nodes;
    }();
    std::unordered_map<NodeType, NodeType> node2root;
    const auto& VisitPrevNode = [&](NodeType node,
                                    const NodeHandlerType& NodeHandler) {
      VisitPrevNodes_(node, [&](NodeType prev_node) {
        if (node2root.count(prev_node) == 0) {
          NodeHandler(prev_node);
        }
      });
    };
    for (NodeType root : dfs_ordered_nodes) {
      if (node2root.count(root) > 0) {
        continue;
      }
      std::vector<NodeType> scc;
      // Use node2root immutably inside dfs visitor.
      DfsVisitor<NodeType> visitor(VisitPrevNode);
      visitor(root, [&](NodeType node) { scc.push_back(node); });
      SccHandler(scc);
      // Update node2root outside dfs visitor.
      for (NodeType node : scc) {
        PADDLE_ENFORCE_EQ(node2root.emplace(node, root).second,
                          true,
                          ::common::errors::AlreadyExists(
                              "Failed to insert the node into node2root. The "
                              "node may already exist."));
      }
    }
  }

 private:
  NodesVisitorType VisitPrevNodes_;
  NodesVisitorType VisitNextNodes_;
};

}  // namespace common
}  // namespace cinn
