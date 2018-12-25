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

#include "paddle/fluid/inference/analysis/analyzer_utils.h"
#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/passes/passes.h"

namespace paddle {
namespace inference {
namespace analysis {

void RunAnalysis(Argument *argument, bool sub_graph_mode) {
  // All the AnalysisPass to run.
  std::vector<std::string> passes({
      "ir_graph_build_pass", "ir_analysis_pass", "subgraph_analysis_pass",
      "ir_params_sync_among_devices_pass",
  });

  for (const auto &pass : passes) {
    VLOG(2) << "Run pass " << pass;
    auto *the_pass = PassRegistry::Global().Retreive(pass);
    // If not compatible with sub-graph analysis, doesn't run this pass.
    if (sub_graph_mode && !the_pass->support_subgraph()) continue;
    the_pass->Run(argument);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
