// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/mir/pass_manager.h"
#include "paddle/fluid/lite/core/mir/ssa_graph.h"

namespace paddle {
namespace lite {

/*
 * lite::Optimizer optimize a program. It utilize the mir passes to analysis the
 * program and export an optimized program.
 */
class Optimizer {
 public:
  void Run(mir::Program&& program, const std::vector<Place>& valid_places,
           const std::vector<std::string>& passes = {}) {
    CHECK(!graph_) << "duplicate optimize found";
    graph_.reset(new mir::SSAGraph);
    graph_->Build(program, valid_places);
    RunPasses();
  }

  // Generate a new program based on the mir graph.
  std::unique_ptr<mir::Program> GenProgram() {
    std::unique_ptr<mir::Program> res;
    return res;
  }

  // Generate C++ code which combines the inference program, model and weights.
  void GenCode(const std::string& code_dir);

  const mir::SSAGraph& ssa_graph() const {
    CHECK(graph_);
    return *graph_;
  }

 protected:
  // Run the default passes registered in the PassManager.
  void RunPasses() { mir::PassManager::Global().Run(graph_); }

  // Specify the passes and run them.
  void RunPasses(std::vector<std::string>& passes);

 private:
  std::unique_ptr<mir::SSAGraph> graph_;
};

}  // namespace lite
}  // namespace paddle
