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

#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/analysis/ir_pass_manager.h"

namespace paddle::inference::analysis {

void IrAnalysisPass::RunImpl(Argument* argument) {
  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  ARGUMENT_CHECK_FIELD(argument, main_program);
  ARGUMENT_CHECK_FIELD(argument, scope);

  auto* the_graph = argument->ReleaseMainGraph();
  auto graph = std::unique_ptr<Graph>(the_graph);

  // Apply passes.
  IRPassManager the_ir_manager(argument);
  graph = the_ir_manager.Apply(std::move(graph));
  PADDLE_ENFORCE_GT(
      graph->Nodes().size(),
      0,
      common::errors::PreconditionNotMet(
          "The graph nodes size should be greater than 0, but got 0"));
  argument->SetMainGraph(graph.release());
  CollectFusionStatis(argument);
}

void IrAnalysisPass::CollectFusionStatis(Argument* argument) {
  if (!argument->main_graph().Has(framework::ir::kFuseStatisAttr)) {
    LOG(INFO) << "argument has no fuse statis";
    return;
  }
  argument->SetFusionStatis(
      argument->main_graph().Get<Argument::fusion_statis_t>(
          framework::ir::kFuseStatisAttr));
}

std::string IrAnalysisPass::repr() const { return "ir_analysis_pass"; }

}  // namespace paddle::inference::analysis
