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

#include "paddle/fluid/inference/analysis/passes/subgraph_analysis_pass.h"
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/subblock_to_graph_pass.h"
#include "paddle/fluid/inference/analysis/analyzer_utils.h"

namespace paddle {
namespace inference {
namespace analysis {
using framework::ir::kSubblockGraphAttr;
using framework::ir::SubblockToGraphPass;
using framework::ir::kParamScopeAttr;

void SubgraphAnalysisPass::RunImpl(Argument* argument) {
  if (!argument->main_graph().Has(kSubblockGraphAttr)) {
    return;
  }

  std::vector<std::string> passes({});

  auto& graphs = argument->main_graph().Get<SubblockToGraphPass::subgraphs_t>(
      kSubblockGraphAttr);
  for (auto& elem : graphs) {
    auto& graph = elem.second;

    Argument arg(*argument);
    // the main_graph, scope, main_program should be set.
    arg.SetMainGraphNotOwned(graph.get());
    arg.SetScopeNotOwned(argument->scope_ptr());
    arg.SetMainProgramNotOwned(
        const_cast<framework::ProgramDesc*>(&graph->program()));
    graph->Set(
        kParamScopeAttr,
        new framework::Scope*(
            argument->main_graph().Get<framework::Scope*>(kParamScopeAttr)));
    // Call the analyzer
    RunAnalysis(&arg, true /*sub_block_mode*/);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
