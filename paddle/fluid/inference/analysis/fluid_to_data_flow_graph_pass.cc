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
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;
using framework::ir::Node;

bool FluidToDataFlowGraphPass::Initialize(Argument *argument) {
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument);
  if (argument->original_program_desc) {
    LOG(WARNING) << "argument's original_program_desc is already set, might "
                    "duplicate called";
  }
  if (!argument->model_program_path) {
    ANALYSIS_ARGUMENT_CHECK_FIELD(argument->model_dir);
    argument->model_program_path.reset(
        new std::string(*argument->model_dir + "/__model__"));
  }
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument->model_program_path);
  auto program = LoadProgramDesc(*argument->model_program_path);
  argument->original_program_desc.reset(
      new framework::proto::ProgramDesc(program));

  desc_ = argument->original_program_desc.get();
  return true;
}

bool FluidToDataFlowGraphPass::Finalize() { return true; }

void FluidToDataFlowGraphPass::Run(Graph *graph) {
  if (!argument_->main_graph) {
    argument_->main_graph.reset(
        new Graph{framework::ProgramDesc(desc_->SerializeAsString())});
  }
  PADDLE_ENFORCE(graph);
  PADDLE_ENFORCE(desc_);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
