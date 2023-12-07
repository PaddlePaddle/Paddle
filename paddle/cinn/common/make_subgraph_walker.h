// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "paddle/cinn/common/make_is_reachable_from_src_predicator.h"
#include "paddle/cinn/common/topo_walker.h"

namespace cinn::common {

template <typename NodeT, typename IterT>
cinn::common::TopoWalker<NodeT> MakeSubgraphWalker(
    const cinn::common::TopoWalker<NodeT>& walker,
    IterT src_begin,
    IterT src_end,
    IterT sink_begin,
    IterT sink_end) {
  cinn::common::TopoWalker<NodeT> reversed_walker(walker.VisitNextNodes,
                                                  walker.VisitPrevNodes);
  auto ReachableToOneSrc =
      cinn::common::MakeIsReachableFromSrcPredicator<NodeT, IterT>(
          walker, src_begin, src_end);
  auto ReachableToOneSink =
      cinn::common::MakeIsReachableFromSrcPredicator<NodeT, IterT>(
          reversed_walker, sink_begin, sink_end);

  auto VisitPrevNodes = [ReachableToOneSrc, ReachableToOneSink, walker](
                            NodeT node,
                            const std::function<void(NodeT)>& Visitor) {
    walker.VisitPrevNodes(node, [&](NodeT in_node) {
      if (ReachableToOneSrc(in_node) && ReachableToOneSink(in_node)) {
        Visitor(in_node);
      }
    });
  };

  auto VisitNextNodes = [ReachableToOneSrc, ReachableToOneSink, walker](
                            NodeT node,
                            const std::function<void(NodeT)>& Visitor) {
    walker.VisitNextNodes(node, [&](NodeT out_node) {
      if (ReachableToOneSrc(out_node) && ReachableToOneSink(out_node)) {
        Visitor(out_node);
      }
    });
  };

  return cinn::common::TopoWalker<NodeT>(VisitPrevNodes, VisitNextNodes);
}

}  // namespace cinn::common
