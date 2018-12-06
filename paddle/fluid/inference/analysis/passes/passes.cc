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

#include "paddle/fluid/inference/analysis/passes/passes.h"
#include "paddle/fluid/inference/analysis/passes/ir_analysis_compose_pass.cc"
#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"
#include "paddle/fluid/inference/analysis/passes/ir_graph_build_pass.h"
#include "paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.h"

namespace paddle {
namespace inference {
namespace analysis {
PassRegistry::PassRegistry() {
  passes_.emplace("ir_analysis_pass",
                  std::unique_ptr<AnalysisPass>(new IrAnalysisPass));
  passes_.emplace("ir_graph_build_pass",
                  std::unique_ptr<AnalysisPass>(new IrGraphBuildPass));
  passes_.emplace("ir_analysis_compose_pass",
                  std::unique_ptr<AnalysisPass>(new IrAnalysisComposePass));
  passes_.emplace(
      "ir_params_sync_among_devices_pass",
      std::unique_ptr<AnalysisPass>(new IrParamsSyncAmongDevicesPass));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
