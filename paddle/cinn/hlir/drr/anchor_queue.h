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
#include <limits>
#include <list>
#include <unordered_map>

#include "glog/logging.h"

#include "cinn/common/bfs_visitor.h"
#include "cinn/hlir/drr/undirected_graph_topo.h"

namespace cinn {
namespace hlir {
namespace drr {

template <typename BigGraphNodeType>
class AnchorQueue {
 public:
  AnchorQueue(const UndirectedGraphTopo<BigGraphNodeType>& big_graph_topo,
              const std::function<bool(BigGraphNodeType anchor_node)>&
                  IsAnchorNodeDeprecated)
      : big_graph_topo_(big_graph_topo),
        IsAnchorNodeDeprecated_(IsAnchorNodeDeprecated) {
    TODO(/*thisjiang*/);
  }

  template <typename BigGraphNodeIter>
  void PushRange(
      BigGraphNodeIter begin,
      BigGraphNodeIter end,
      int radius,
      const std::function<bool(BigGraphNodeType node)>& IsAnchorPredicator) {
    struct Depth {
      int value;
      Depth() : value(std::numeric_limits<int>::max()) {}
      Depth(int init_value) : value(init_value) {}
    };
    std::unordered_map<BigGraphNodeType, Depth> node2depth{};
    for (BigGraphNodeIter it = begin; it != end; ++it) {
      node2depth[*it] = Depth(0);
    }
    const auto& WalkNextNode =
        [&](BigGraphNodeType node,
            const std::function<void(BigGraphNodeType)>& VisitNode) {
          if (node2depth[node].value <= radius) {
            big_graph_topo_.WalkNextNodes(node, VisitNode);
          }
        };
    common::BfsVisitor<BigGraphNodeType> bfs_visitor(WalkNextNode);
    bfs_visitor(begin, end, [&](BigGraphNodeType node) {
      if (IsAnchorPredicator(node)) {
        anchor_node_queue_.push_back(node);
      }
      WalkNextNode(node, [&](BigGraphNodeType next_node) {
        node2depth[next_node].value =
            std::min(node2depth[next_node].value, node2depth[node].value + 1);
      });
    });
  }

  void PushByRadius(
      BigGraphNodeType center_node,
      int radius,
      const std::function<bool(BigGraphNodeType node)>& IsAnchorPredicator) {
    std::array<BigGraphNodeType, 1> start_nodes{center_node};
    PushRange(
        start_nodes.begin(), start_nodes.end(), radius, IsAnchorPredicator);
  }

  bool Empty() const {
    for (const auto& node : anchor_node_queue_) {
      if (!IsAnchorNodeDeprecated(node)) {
        return false;
      }
    }
    return true;
  }

  BigGraphNodeType Pop() {
    while (true) {
      const auto& anchor_node = anchor_node_queue_.front();
      anchor_node_queue_.pop_front();

      if (!IsAnchorNodeDeprecated(anchor_node)) {
        return anchor_node;
      }
    }
    LOG(FATAL) << "No non deprecated node found";
  }

 private:
  const UndirectedGraphTopo<BigGraphNodeType> big_graph_topo_;
  const std::function<bool(BigGraphNodeType anchor_node)>
      IsAnchorNodeDeprecated_;

  std::list<BigGraphNodeType> anchor_node_queue_;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
