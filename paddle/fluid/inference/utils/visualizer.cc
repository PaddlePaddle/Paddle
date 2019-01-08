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

#include "paddle/fluid/inference/utils/visualizer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <memory>
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"
#include "paddle/fluid/platform/init.h"

DEFINE_string(model_dir, "", "model directory");
DEFINE_string(model_program_path, "", "model program path");
DEFINE_string(model_params_path, "", "model params path");

using paddle::inference::analysis::Argument;

namespace paddle {
namespace inference {
namespace utils {

void Visualizer::SetArgument(Argument *argument) { argument_ = argument; }

bool Visualizer::Run() {
  paddle::framework::InitDevices(false);
  paddle::inference::analysis::Analyzer().Run(argument_);
  return true;
}

}  // namespace utils
}  // namespace inference
}  // namespace paddle

// Generate a dot file describing the structure of graph.
// To use this tool, run command: ./visualizer [options...]
// Options:
//     --model_dir: the directory of model
//     --model_program_path: the path of program
//     --model_params_path: the path of params
int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  paddle::inference::analysis::Argument argument;
  argument.SetUseGPU(false);
  argument.SetUseTensorRT(false);

  if (FLAGS_model_dir.empty()) {
    if (FLAGS_model_program_path.empty() || FLAGS_model_params_path.empty()) {
      LOG(ERROR) << "Please set model_dir"
                    " or model_program_path and model_params_path";
      return -1;
    } else {
      argument.SetModelProgramPath(FLAGS_model_program_path);
      argument.SetModelParamsPath(FLAGS_model_params_path);
    }
  } else {
    argument.SetModelDir(FLAGS_model_dir);
  }

  // Only 1 pass, default filename is 0_ir_origin.dot
  // For more details, looking for paddle::inference::analysis::IRPassManager
  argument.SetIrAnalysisPasses({"infer_clean_graph_pass", "graph_viz_pass"});

  std::unique_ptr<paddle::framework::Scope> scope{
      new paddle::framework::Scope()};
  argument.SetScopeNotOwned(
      const_cast<paddle::framework::Scope *>(scope.get()));

  paddle::inference::utils::Visualizer visualizer;
  visualizer.SetArgument(&argument);
  visualizer.Run();

  return 0;
}

USE_PASS(infer_clean_graph_pass);
USE_PASS(graph_viz_pass);
USE_PASS(graph_to_program_pass);
