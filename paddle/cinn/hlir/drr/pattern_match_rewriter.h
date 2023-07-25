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

#include "paddle/cinn/hlir/drr/source_pattern_graph.h"

namespace cinn {
namespace hlir {
namespace drr {

class PatternMatchRewriter {
 public:
  PatternMatchRewriter(const SourcePatternGraph* source_pattern_graph,
                       const std::vector<Constrain*>& constraints,
                       const ResultPatternGraph* result_pattern_graph)
      : source_pattern_graph_(source_pattern_graph),
        constraints_(constraints),
        result_pattern_graph_(result_pattern_graph) {}

  bool Apply(ir::Program* program_) {
    // step 1
    auto result = MatchPattern(source_pattern_graph_, program);
    // step 2
    if (ConstrainsCheck(result, constraints_)) {
      // step 3
      RewritePattern(result_pattern_graph_, result);
      return true;
    }
    return false;
  }

 private:
  const SourcePatternGraph* source_pattern_graph_;
  const std::vector<Constrain*>& constraints_;
  const ResultPatternGraph* result_pattern_graph_;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
