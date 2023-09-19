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

/*
 * This file defines IRPassManager, it helps control the passes in IR. Inference
 * phrase will load the model program and parameters from disk, that is quite
 * different from the training phase.
 * This manager will control the Passes and make the passes in IR work smoothly
 * for inference.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {
namespace analysis {
using framework::ProgramDesc;
using framework::ir::Graph;
using framework::ir::Pass;

class IRPassManager final {
 public:
  explicit IRPassManager(Argument *argument);

  std::unique_ptr<Graph> Apply(std::unique_ptr<Graph> graph);

 private:
  void CreatePasses(Argument *argument, const std::vector<std::string> &passes);

  std::vector<std::unique_ptr<Pass>> passes_;
  bool disable_logs_{false};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
