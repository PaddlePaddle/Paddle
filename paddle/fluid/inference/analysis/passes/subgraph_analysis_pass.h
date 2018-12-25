// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include "paddle/fluid/inference/analysis/analysis_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Analysis the subgraph generated from subblock.
 * It works with the operators with sub-blocks, such as WhileOp, by executing
 * the existing passes, it will fuse some nodes in the sub-graphs.
 */
class SubgraphAnalysisPass : public AnalysisPass {
 public:
  SubgraphAnalysisPass() {
    // Nested sub-graph analysis is too complex, not support yet.
    SetSupportSubgraph(false);
  }

  void RunImpl(Argument *argument) override;

  // Copy the sub-graph's program desc to the main_graph's program desc.
  void TransferAnalyzedProgramToBlockDesc(
      const framework::ProgramDesc &subgraph_program,
      framework::BlockDesc *block_in_main_graph,
      framework::ProgramDesc *main_program);

  void DrawGraph(Argument *argument);

  std::string repr() const override { return "subgraph-analysis-pass"; }
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
