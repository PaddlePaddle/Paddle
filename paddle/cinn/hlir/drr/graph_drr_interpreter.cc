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

#include "cinn/hlir/drr/graph_drr_interpreter.h"
#include <unordered_set>

#include "cinn/hlir/drr/anchor_queue.h"
#include "cinn/hlir/drr/source_pattern.h"
#include "cinn/hlir/drr/visit_status.h"
#include "cinn/hlir/framework/node.h"

namespace cinn {
namespace hlir {
namespace drr {

void GraphDrrInterpreter::operator()(
    const SourcePattern& source_pattern,
    const ResultPattern& result_pattern) const {
  AnchorQueue<framework::Node> anchor_queue;
  TODO(/*initial anchor queue here*/);

  const auto& ReplaceSourceGraphToResultGraph =
      [&](const MatchContext& match_context) {
        std::unordered_set<const framework::Node*> effected_nodes{};
        TODO(/*replace the matched source graph to result graph by match context*/);
        return effected_nodes;
      };

  // visit anchor queue
  GraphDrrMatcher graph_drr_matcher(*source_pattern.source_pattern_graph(),
                                    *graph_);
  anchor_queue.DynamicallyWalkAnchorNode([&](const framework::Node* node) {
    std::unordered_set<const framework::Node*> effected_nodes{};
    graph_drr_matcher(
        node, [&](const MatchContext& match_context) -> VisitStatus {
          if (source_pattern.AllConstrainsMatched(match_context)) {
            effected_nodes = ReplaceSourceGraphToResultGraph(match_context);
            return VisitStatus::kBreak;
          }
          return VisitStatus::kContinue;
        });
    return effected_nodes;
  });
}

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
