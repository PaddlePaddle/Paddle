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
#include "paddle/fluid/lite/core/mir/generate_program_pass.h"
#include "paddle/fluid/lite/core/mir/io_complement_pass.h"
#include "paddle/fluid/lite/core/mir/pass_manager.h"
#include "paddle/fluid/lite/core/mir/ssa_graph.h"
#include "paddle/fluid/lite/core/mir/static_kernel_pick_pass.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/types.h"

namespace paddle {
namespace lite {

/*
 * lite::Optimizer optimize a program. It utilize the mir passes to analysis the
 * program and export an optimized program.
 */
class Optimizer {
 public:
  void Run(Program&& program, const std::vector<Place>& valid_places,
           core::KernelPickFactor kernel_pick_factor,
           const std::vector<std::string>& passes = {}) {
    valid_places_ = valid_places;
    CHECK(!valid_places.empty()) << "At least one valid_place should be set";
    CHECK(!graph_) << "duplicate optimize found";
    graph_.reset(new mir::SSAGraph);
    graph_->Build(program, valid_places);
    SpecifyKernelPickTactic(kernel_pick_factor);
    // InitIoComplement();
    RunPasses();
    exec_scope_ = program.exec_scope;
  }

  void KernelPickPreferPlace(const Place& place) {
    auto* pass = mir::PassManager::Global().LookUp<mir::StaticKernelPickPass>(
        "static_kernel_pick_pass");
    CHECK(pass);
    pass->SetPreferPlace(place);
  }

  // Generate a new program based on the mir graph.
  std::unique_ptr<RuntimeProgram> GenRuntimeProgram() {
    LOG(INFO) << "generate program";
    std::unique_ptr<Program> res;
    auto pass = mir::PassManager::Global().LookUp<mir::GenerateProgramPass>(
        "generate_program_pass");
    pass->Apply(graph_);
    auto program = pass->GenProgram();
    CHECK(exec_scope_);
    program->set_exec_scope(exec_scope_);
    return program;
  }

  void InitIoComplement() {
    auto* pass = mir::PassManager::Global().LookUp<mir::IoComplementPass>(
        "io_complement_pass");
    CHECK(pass);
    CHECK(!valid_places_.empty());
    LOG(INFO) << "valid_places.size " << valid_places_.size();
    pass->SetValidPlaces(valid_places_);
  }

  // Generate C++ code which combines the inference program, model and weights.
  void GenCode(const std::string& code_dir);

  const mir::SSAGraph& ssa_graph() const {
    CHECK(graph_);
    return *graph_;
  }

 protected:
  void SpecifyKernelPickTactic(core::KernelPickFactor factor);

  // Run the default passes registered in the PassManager.
  void RunPasses();

  // Specify the passes and run them.
  void RunPasses(std::vector<std::string>& passes);

 private:
  std::unique_ptr<mir::SSAGraph> graph_;
  std::vector<Place> valid_places_;
  lite::Scope* exec_scope_{};
};

}  // namespace lite
}  // namespace paddle
