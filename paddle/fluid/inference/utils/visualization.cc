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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <memory>
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"
#include "paddle/fluid/platform/init.h"

DEFINE_string(model_dir, "", "model path");
DEFINE_string(model_program, "", "model program path");
DEFINE_string(model_params, "", "model params path");

USE_PASS(graph_viz_pass);
USE_PASS(graph_to_program_pass);

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  paddle::inference::analysis::Argument argument;
  argument.SetUseGPU(false);
  argument.SetUseTensorRT(false);

  if (FLAGS_model_dir.empty()) {
    if (FLAGS_model_program.empty() || FLAGS_model_params.empty()) {
      LOG(ERROR) << "Please set model_dir or model_program and model_params";
      return -1;
    } else {
      argument.SetModelProgramPath(FLAGS_model_program);
      argument.SetModelParamsPath(FLAGS_model_params);
    }
  } else {
    argument.SetModelDir(FLAGS_model_dir);
  }

  argument.SetIrAnalysisPasses({"graph_viz_pass"});
  paddle::framework::InitDevices(false);
  std::unique_ptr<paddle::framework::Scope> scope{
      new paddle::framework::Scope()};
  argument.SetScopeNotOwned(
      const_cast<paddle::framework::Scope *>(scope.get()));

  paddle::inference::analysis::Analyzer().Run(&argument);

  return 0;
}
