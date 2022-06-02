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
#include "paddle/fluid/platform/profiler/event_tracing.h"

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
  // NOTE(zhiqiu): it is needed to sync the variables in scope to
  // variable_scope, since the some variable only exists in scope.
  // For example, 'lod_tensor_blocking_queue_0' used in dataloader.
  // These variables may be created in scope, and it is not existed as
  // variable in program.
  if (scope) {
    const std::string blocking_queue_prefix = "lod_tensor_blocking_queue";
    auto vars = scope->LocalVarNames();
    for (const auto& name : vars) {
      if (name.find(blocking_queue_prefix) != std::string::npos) {
        if (!global_scope_.HasVar(name)) {
          auto* v = scope->Var(name);
          VLOG(4) << "Sync Variable from scope to variable scope: " << name;
          global_scope_.AddVar(name, *v);
        }
      }
    }
  }

  // NOTE(zhiqiu): for startup_program, initialize scope and run once
  // if startup_program is empty, the scope is initialize during first run
  if (startup_prog.Block(0).AllOps().size() > 0) {
    VLOG(4) << "Run startup program";
    // init scope
    BuildVariableScope(startup_prog, &global_scope_);
    std::vector<paddle::framework::OpFuncNode> vec_func_list;
    // No need to use_local_scope for startup_program, its variables are
    // persistable
    paddle::framework::interpreter::build_op_func_list(
        place_, startup_prog.Block(0), &vec_func_list, &global_scope_, false);
  }
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors,
    const std::vector<std::string>& fetch_names) {
  platform::RecordEvent record_event("StandaloneExecutor::run",
                                     platform::TracerEventType::UserDefined, 1);

  auto core = GetInterpreterCore(feed_names, fetch_names, true);

  return core->Run(feed_names, feed_tensors);
}

paddle::framework::FetchList StandaloneExecutor::Run(
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names) {
  platform::RecordEvent record_event("StandaloneExecutor::run",
                                     platform::TracerEventType::UserDefined, 1);

  auto core = GetInterpreterCore(feed_names, fetch_names, false);
  VLOG(4) << "StandaloneExecutor: " << this << ", InterpreterCore: " << core;
  return core->Run(feed_names);
}

framework::interpreter::CostInfo StandaloneExecutor::DryRun(
    const std::vector<std::string>& feed_names,
    const std::vector<framework::LoDTensor>& feed_tensors) {
  auto core = GetInterpreterCore(feed_names, {}, true);

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
      VLOG(4) << "Create variable from startup_prog: "
              << var->Proto()->SerializeAsString();
      var_scope->AddVar(var->Name(), var);
    }
  }
}

std::shared_ptr<InterpreterCore> StandaloneExecutor::GetInterpreterCore(
    const std::vector<std::string>& feed_names,
    const std::vector<std::string>& fetch_names, bool add_fetch_op) {
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
    VLOG(3) << "create interpreter_core for " << oss.str() << " on place "
            << place_;
    VLOG(3) << "add fetch op: " << add_fetch_op;
    std::shared_ptr<InterpreterCore> core = nullptr;
    if (add_fetch_op) {
      // NOTE(Aurelius84): `add_fetch` will modify BlockDesc, so we should copy
      // a
      // new program.
      auto new_prog = std::make_shared<framework::ProgramDesc>(main_prog_);
      auto* block = new_prog->MutableBlock(0);
      interpreter::add_fetch(fetch_names, block);

      core = std::make_shared<InterpreterCore>(place_, *block, &global_scope_);
      core->SetCopyProgram(new_prog);
    } else {
      core = std::make_shared<InterpreterCore>(place_, main_prog_.Block(0),
                                               &global_scope_);
    }
    interpretercores_.emplace(oss.str(), core);
    return core;
  } else {
    return iter->second;
  }
}

}  // namespace framework
}  // namespace paddle
