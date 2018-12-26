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
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/subblock_to_graph_pass.h"
#include "paddle/fluid/inference/analysis/analyzer_utils.h"
#include "subgraph_analysis_pass.h"

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
    auto* graph_raw_ptr = graph.release();
    arg.SetMainGraph(graph_raw_ptr);
    arg.SetScopeNotOwned(argument->scope_ptr());
    arg.SetMainProgram(new framework::ProgramDesc(graph_raw_ptr->program()));
    arg.main_graph().Set(
        kParamScopeAttr,
        new framework::Scope*(
            argument->main_graph().Get<framework::Scope*>(kParamScopeAttr)));

    // Call the analyzer
    RunAnalysis(&arg, true /*sub_block_mode*/);

    graph.reset(arg.ReleaseMainGraph());

    // Set the optimized program directly to the Op with sub-blocks.

    LOG(INFO) << "elem " << elem.first->Name();
    auto* sub_block_in_main_graph = boost::get<framework::BlockDesc*>(
        elem.first->Op()->GetAttr("sub_block"));
    framework::ProgramDesc program(arg.ir_analyzed_program());
    TransferAnalyzedProgramToBlockDesc(program, sub_block_in_main_graph,
                                       argument->main_program_ptr());
  }

  // A temporary way to display the structure after sub-graph analysis.
  // Not compatible with the drawing logic in IrPassManager.
  DrawGraph(argument);
}

void SubgraphAnalysisPass::TransferAnalyzedProgramToBlockDesc(
    const framework::ProgramDesc& subgraph_program,
    framework::BlockDesc* block_in_main_graph,
    framework::ProgramDesc* main_program) {
  const auto& block = subgraph_program.Block(0);
  // Copy operators to target block.
  block_in_main_graph->Proto()->mutable_ops()->Clear();
  block_in_main_graph->Flush();
  for (size_t i = 0; i < block.OpSize(); i++) {
    *block_in_main_graph->AppendOp()->Proto() = *block.Op(i)->Proto();
  }

  // Copy temporary vars.
  block_in_main_graph->Proto()->mutable_vars()->Clear();
  block_in_main_graph->Flush();
  for (auto& var_name : block.LocalVarNames()) {
    auto* var = block.FindVar(var_name);
    if (!var->Persistable()) {
      *block_in_main_graph->Var(var_name)->Proto() = *var->Proto();
    }
  }

  // No need to copy parameters to main program, because the WhileOp(s) has
  // dependency on them.
}

void SubgraphAnalysisPass::DrawGraph(Argument* argument) {
  framework::ir::GraphVizPass pass;
  pass.Set("graph_viz_path", new std::string("subgraph_analysis_pass.dot"));
  argument->SetMainGraph(
      pass.Apply(std::unique_ptr<Graph>(argument->ReleaseMainGraph()))
          .release());
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
