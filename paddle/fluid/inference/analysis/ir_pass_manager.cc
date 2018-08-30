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

#include "paddle/fluid/inference/analysis/ir_pass_manager.h"
#include <string>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace inference {
namespace analysis {

IRPassManager::IRPassManager(const ProgramDesc &program,
                             framework::Scope *scope)
    : program_(program) {
  graph_.reset(new framework::ir::Graph(program));
  if (scope) graph_->Set("param_scope", new framework::Scope *(scope));
}

void IRPassManager::Apply(const std::vector<std::string> &passes) {
  // Apply all the passes
  std::string pre_pass;
  for (const std::string &pass_name : passes) {
    LOG(WARNING) << "Running IR pass [" << pass_name << "]";
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_name);
    if (pass_name == "graph_viz_pass") {
      std::string dot_file_path =
          "ir_" + (pre_pass.empty() ? "origin" : pre_pass) + ".dot";
      pass->Set("graph_viz_path", new std::string(std::move(dot_file_path)));
    }
    graph_ = pass->Apply(std::move(graph_));
    pre_pass = pass_name;
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
