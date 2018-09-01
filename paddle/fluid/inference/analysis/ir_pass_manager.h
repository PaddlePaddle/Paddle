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

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace inference {
namespace analysis {
using framework::ProgramDesc;

class IRPassManager final {
 public:
  IRPassManager(const ProgramDesc &program, framework::Scope *scope);

  void Apply(const std::vector<std::string> &passes);

  framework::ir::Graph &graph() const { return *graph_; }

 private:
  std::unique_ptr<framework::ir::Graph> graph_;
  ProgramDesc program_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
