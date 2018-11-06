/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <string>
#include <vector>

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

bool FluidToDataFlowGraphPass::Initialize(Argument *argument) {
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument);
  if (argument->origin_program_desc) {
    LOG(WARNING) << "argument's origin_program_desc is already set, might "
                    "duplicate called";
  }
  if (!argument->fluid_model_program_path) {
    ANALYSIS_ARGUMENT_CHECK_FIELD(argument->fluid_model_dir);
    argument->fluid_model_program_path.reset(
        new std::string(*argument->fluid_model_dir + "/__model__"));
  }
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument->fluid_model_program_path);
  auto program = LoadProgramDesc(*argument->fluid_model_program_path);
  argument->origin_program_desc.reset(
      new framework::proto::ProgramDesc(program));

  if (!argument->main_dfg) {
    argument->main_dfg.reset(new DataFlowGraph);
  }
  desc_ = argument->origin_program_desc.get();
  return true;
}

bool FluidToDataFlowGraphPass::Finalize() { return true; }

void FluidToDataFlowGraphPass::Run(DataFlowGraph *graph) {
  PADDLE_ENFORCE(graph);
  PADDLE_ENFORCE(desc_);
  graph->Build(*desc_);
}

namespace {
class DFG_DebuggerPass : public DFG_GraphvizDrawPass {
 public:
  using Config = DFG_GraphvizDrawPass::Config;
  explicit DFG_DebuggerPass(const Config &config)
      : DFG_GraphvizDrawPass(config) {}
  std::string repr() const override { return "fluid-to-dfg-debuger-pass"; }
  bool Finalize() override { return true; }
};
}

AnalysisPass *FluidToDataFlowGraphPass::CreateGraphvizDebugerPass() const {
  return new DFG_DebuggerPass(DFG_GraphvizDrawPass::Config(
      FLAGS_IA_graphviz_log_root, "fluid-to-dfg-debuger"));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
