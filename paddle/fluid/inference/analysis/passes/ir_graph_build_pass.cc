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

#include "paddle/fluid/inference/analysis/passes/ir_graph_build_pass.h"
#include <paddle/fluid/framework/ir/fuse_pass_base.h>
#include <string>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {

extern void ReadBinaryFile(const std::string &filename, std::string *contents);

namespace analysis {

void IrGraphBuildPass::RunImpl(Argument *argument) {
  if (!argument->scope()) {
    argument->SetScope(std::unique_ptr<framework::Scope>(new framework::Scope));
  }

  if (argument->model_dir()) {
    auto program = LoadModel(*argument->model_dir(), argument->scope());
    argument->SetMainProgram(std::move(program));
  } else if (argument->model_program_path() && argument->model_params_path()) {
    auto program = LoadModel(*argument->model_program_path(),
                             *argument->model_params_path(), argument->scope());
    argument->SetMainProgram(std::move(program));
  } else {
    PADDLE_THROW(
        "either model_dir or (program path and parameter path) should be set.");
  }

  auto graph = std::unique_ptr<Graph>(new Graph(*argument->main_program()));
  LOG(INFO) << "Load " << graph->Nodes().size() << " nodes";
  argument->SetMainGraph(std::move(graph));
  argument->main_graph()->Set(framework::ir::kParamScopeAttr,
                              new framework::Scope *(argument->scope()));
}

std::unique_ptr<framework::ProgramDesc> IrGraphBuildPass::LoadModel(
    const std::string &path, framework::Scope *scope) {
  platform::CPUPlace place;
  framework::Executor exe(place);
  return Load(&exe, scope, path);
}

std::unique_ptr<framework::ProgramDesc> IrGraphBuildPass::LoadModel(
    const std::string &program_path, const std::string &params_path,
    framework::Scope *scope) {
  platform::CPUPlace place;
  framework::Executor exe(place);
  return Load(&exe, scope, program_path, params_path);
}

std::string IrGraphBuildPass::repr() const { return "ir-graph-build-pass"; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
