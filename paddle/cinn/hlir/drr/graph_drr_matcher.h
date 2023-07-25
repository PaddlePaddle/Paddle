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
#include <memory>

#include "cinn/hlir/drr/drr_pass_context.h"
#include "cinn/hlir/drr/input_sorted_graph_matcher.h"
#include "cinn/hlir/drr/visit_status.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace hlir {
namespace drr {

class SourcePatternGraph;
class MatchContextImpl;

class GraphDrrMatcher final {
 public:
  GraphDrrMatcher(const GraphDrrMatcher&) = delete;
  GraphDrrMatcher(GraphDrrMatcher&&) = delete;

  using SmallGraphNodeType = OpCall*;
  using BigGraphNodeType = framework::Node*;

  GraphDrrMatcher(const SourcePatternGraph& small_graph,
                  const framework::Graph& big_graph)
      : small_graph_(small_graph), big_graph_(big_graph) {
    TODO(/*thisjiang*/);
  }

  void WalkMatchContext(
      const BigGraphNodeType anchor_node,
      const std::function<VisitStatus(const MatchContext& match_context)>&
          VisitMatchContext) const;

 private:
  void WalkMatchContextImpl(
      const BigGraphNodeType anchor_node,
      const std::function<VisitStatus(
          std::unique_ptr<const MatchContextImpl>&& match_context_impl)>&
          VisitMatchContextImpl) const {
    TODO(/*thisjiang*/);
  }

  const SourcePatternGraph& small_graph_;
  const framework::Graph& big_graph_;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
