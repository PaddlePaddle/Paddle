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

#include "paddle/fluid/inference/analysis/passes/adjust_cudnn_workspace_size_pass.h"
#include "paddle/fluid/inference/analysis/passes/inference_op_replace_pass.h"
#include "paddle/fluid/inference/analysis/passes/ir_analysis_pass.h"
#include "paddle/fluid/inference/analysis/passes/ir_graph_build_pass.h"
#include "paddle/fluid/inference/analysis/passes/ir_graph_to_program_pass.h"
#include "paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"
#include "paddle/fluid/inference/analysis/passes/save_optimized_model_pass.h"

namespace paddle::inference::analysis {

PassRegistry::PassRegistry() {  // NOLINT
  // Register manually to avoid the trivial `USE_OP` like macro for easier use
  // and link.
  passes_.emplace("ir_analysis_pass", std::make_unique<IrAnalysisPass>());
  passes_.emplace("ir_graph_build_pass", std::make_unique<IrGraphBuildPass>());
  passes_.emplace("save_optimized_model_pass",
                  std::make_unique<SaveOptimizedModelPass>());
  passes_.emplace("memory_optimize_pass",
                  std::make_unique<MemoryOptimizePass>());
  passes_.emplace("ir_params_sync_among_devices_pass",
                  std::make_unique<IrParamsSyncAmongDevicesPass>());
  passes_.emplace("adjust_cudnn_workspace_size_pass",
                  std::make_unique<AdjustCudnnWorkSpacePass>());
  passes_.emplace("inference_op_replace_pass",
                  std::make_unique<InferenceOpReplacePass>());
  passes_.emplace("ir_graph_to_program_pass",
                  std::make_unique<IrGraphToProgramPass>());
}

}  // namespace paddle::inference::analysis
