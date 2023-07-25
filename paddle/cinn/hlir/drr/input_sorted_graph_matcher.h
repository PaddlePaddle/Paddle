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
#include <functional>
#include <unordered_map>
#include <vector>

#include "cinn/hlir/drr/graph_topo.h"
#include "cinn/hlir/drr/match_context.h"
#include "cinn/hlir/drr/sinks_node_matcher.h"

namespace cinn {
namespace hlir {
namespace drr {

template <typename SmallGraphNodeType, typename BigGraphNodeType>
class InputSortedGraphMatcher final {
 public:
  using MatchMap = std::unordered_map<SmallGraphNodeType, BigGraphNodeType>;

  InputSortedGraphMatcher(
      const GraphTopo<SmallGraphNodeType>& small_graph_topo,
      SmallGraphNodeType small_graph_start,
      const GraphTopo<BigGraphNodeType>& big_graph_topo,
      const std::function<bool(SmallGraphNodeType small_node,
                               BigGraphNodeType big_node)>& NodeMatchPredicator)
      : small_graph_topo_(small_graph_topo),
        big_graph_topo_(big_graph_topo),
        NodeMatchPredicator_(NodeMatchPredicator),
        SinksNodeMatcher_(small_graph_topo,
                          small_graph_start,
                          big_graph_topo,
                          NodeMatchPredicator) {
    TODO(/*thisjiang*/);
  }

  void WalkMatchedGraph(
      BigGraphNodeType anchor_node,
      const std::function<VisitStatus(const MatchMap&)>& VisitMatchedResult) {
    TODO(/*thisjiang*/);
  }

 private:
  void WalkMatchGraphFromSinks(
      const typename SinksNodeMatcher<SmallGraphNodeType,
                                      BigGraphNodeType>::NodeMatchMap&
          sinks_match_map,
      BigGraphNodeType anchor_node,
      const std::function<VisitStatus(const MatchMap&)>& VisitMatchedResult)
      const;

  const GraphTopo<SmallGraphNodeType> small_graph_topo_;
  const GraphTopo<BigGraphNodeType> big_graph_topo_;
  const std::function<bool(SmallGraphNodeType small_node,
                           BigGraphNodeType big_node)>
      NodeMatchPredicator_;
  SinksNodeMatcher<SmallGraphNodeType, BigGraphNodeType> SinksNodeMatcher_;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
