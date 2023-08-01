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

#include "paddle/cinn/common/bfs_walker.h"

namespace cinn {
namespace common {

template <typename NodeType>
class IsReachablePredicator final {
 public:
  IsReachablePredicator(const IsReachablePredicator&) = delete;
  IsReachablePredicator(IsReachablePredicator&&) = delete;

  using NodeHandlerType = std::function<void(NodeType)>;
  using NodesVisitorType =
      std::function<void(NodeType, const NodeHandlerType&)>;
  using NodeDepthGetterType = std::function<size_t(NodeType)>;

  IsReachablePredicator(const NodeDepthGetterType& MinDepth4Node,
                        const NodeDepthGetterType& MaxDepth4Node,
                        const NodesVisitorType& VisitNextNodes)
      : MinDepth4Node_(MinDepth4Node),
        MaxDepth4Node_(MaxDepth4Node),
        VisitNextNodes_(VisitNextNodes) {}

  bool operator()(NodeType src,
                  NodeType dst,
                  const NodeHandlerType& HandleVisited) const {
    const size_t dst_max_depth = MaxDepth4Node_(dst);
    bool detect_reachable = false;
    BfsWalker<NodeType> bfs_walker(
        [&](NodeType node, const NodeHandlerType& Handler) {
          VisitNextNodes_(node, [&](NodeType out_node) {
            if (dst_max_depth < MinDepth4Node_(out_node)) {
              // Pruned.
              // Do nothing.
            } else if (detect_reachable) {
              // Pruned.
              // Reachability is detected.
            } else {
              Handler(out_node);
            }
          });
        });
    std::array<NodeType, 1> starts{src};
    bfs_walker(starts.begin(), starts.end(), [&](NodeType node) {
      HandleVisited(node);
      if (node == dst) {
        detect_reachable = true;
      }
    });
    return detect_reachable;
  }

 private:
  NodeDepthGetterType MinDepth4Node_;
  NodeDepthGetterType MaxDepth4Node_;
  NodesVisitorType VisitNextNodes_;
};

}  // namespace common
}  // namespace cinn
