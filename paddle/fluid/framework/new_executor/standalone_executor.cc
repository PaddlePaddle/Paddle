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

namespace paddle {
namespace framework {
StandaloneExecutor::StandaloneExecutor(const platform::Place& place,
                                       const ProgramDesc& startup_prog,
                                       const ProgramDesc& main_prog,
                                       Scope* scope)
    : place_(place),
      startup_prog_(startup_prog),
      main_prog_(main_prog),
      outer_scope_(scope) {
  paddle::framework::InitDevices();

  // init scope
  BuildVariableOuterScope(startup_prog, &global_scope_, scope);

  if (outer_scope_ != nullptr) {
    auto name_list = outer_scope_->LocalVarNames();
    for (auto name : name_list) {
      auto v = outer_scope_->Var(name);
      if (global_scope_.name2id.find(name) == global_scope_.name2id.end()) {
        global_scope_.name2id[name] = global_scope_.var_list.size();
      }

      global_scope_.var_list.push_back(v);
    }
  }

  // run startup program
  std::vector<paddle::framework::OpFuncNode> vec_func_list;
  std::vector<paddle::framework::OperatorBase*> op_list;
  InterpreterCore::BuildOpFuncList(place_, startup_prog, &op_list,
                                   &vec_func_list, &global_scope_);
}

int StandaloneExecutor::Run(const std::vector<std::string>& feed_names,
                            const std::vector<framework::Tensor>& feed_tensors,
                            const std::vector<std::string>& fetch_names,
                            std::vector<framework::Tensor>* fetch_tensors) {
  auto core = GetInterpreterCore(feed_names, fetch_names);

  core->Run(feed_tensors, fetch_tensors);

  return 0;
}

void StandaloneExecutor::BuildVariableOuterScope(
    const framework::ProgramDesc& pdesc, VariableScope* var_scope,
    Scope* outer_scope) {
  auto& global_block = pdesc.Block(0);

  for (auto& var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    if (var_scope->name2id.find(var->Name()) == var_scope->name2id.end()) {
      var_scope->name2id[var->Name()] = var_scope->var_list.size();
      auto v = outer_scope->Var(var->Name());
      InitializeVariable(v, var->GetType());
      var_scope->var_list.push_back(v);
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
    auto core = std::make_shared<InterpreterCore>(
        place_, main_prog_, &global_scope_, feed_names, fetch_names);
    interpretercores_.emplace(oss.str(), core);
    return core;
  } else {
    return iter->second;
  }
}

}  // namespace framework
}  // namespace paddle
