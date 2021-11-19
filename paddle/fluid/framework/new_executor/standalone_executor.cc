// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/new_executor/standalone_executor.h"
#include "paddle/fluid/framework/new_executor/interpretercore_util.h"

namespace paddle {
namespace framework {
StandaloneExecutor::StandaloneExecutor(const platform::Place& place,
                                       const ProgramDesc& startup_prog,
                                       const ProgramDesc& main_prog,
                                       Scope* scope)
    : place_(place),
      startup_prog_(startup_prog),
      main_prog_(main_prog),
      global_scope_(VariableScope(scope)) {
  // init scope
  BuildVariableScope(startup_prog, &global_scope_);

  if (scope != nullptr) {
    auto name_list = scope->LocalVarNames();
    for (auto name : name_list) {
      auto v = scope->Var(name);
      if (!global_scope_.HasVar(name)) {
        global_scope_.AddVar(name, *v);
      }
    }
  }

  // run startup program
  std::vector<paddle::framework::OpFuncNode> vec_func_list;
  paddle::framework::interpreter::build_op_func_list(
      place_, startup_prog.Block(0), &vec_func_list, &global_scope_);
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors,
    const std::vector<std::string>& fetch_names) {
  auto core = GetInterpreterCore(feed_names, fetch_names);

  return core->Run(feed_names, feed_tensors);
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names) {
  auto core = GetInterpreterCore(feed_names, fetch_names);

  return core->Run();
}

framework::interpreter::CostInfo StandaloneExecutor::DryRun(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors) {
  auto core = GetInterpreterCore(feed_names, {});

  return core->DryRun(feed_names, feed_tensors);
}

void StandaloneExecutor::BuildVariableScope(const framework::ProgramDesc& pdesc,
                                            VariableScope* var_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    if (!var_scope->HasVar(var->Name())) {
      var_scope->AddVar(var->Name(), var);
    }
  }
}

std::shared_ptr<InterpreterCore> StandaloneExecutor::GetInterpreterCore(
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names) {
  std::ostringstream oss;
  oss << "feed:";
  for (auto& feedname : feed_names) {
    oss << feedname << ",";
  }
  oss << "fetch:";
  for (auto& fetchname : fetch_names) {
    oss << fetchname << ",";
  }

  auto iter = interpretercores_.find(oss.str());

  if (iter == interpretercores_.end()) {
    VLOG(3) << "create interpreter_core for " << oss.str();
    // NOTE(Aurelius84): `add_fetch` will modify BlockDesc, so we should copy a
    // new program.
    auto new_prog = std::make_shared<framework::ProgramDesc>(main_prog_);
    auto* block = new_prog->MutableBlock(0);
    interpreter::add_fetch(fetch_names, block);

    auto core =
        std::make_shared<InterpreterCore>(place_, *block, &global_scope_);
    programs_.emplace(oss.str(), new_prog);
    interpretercores_.emplace(oss.str(), core);
    return core;
  } else {
    return iter->second;
  }
}

}  // namespace framework
}  // namespace paddle
