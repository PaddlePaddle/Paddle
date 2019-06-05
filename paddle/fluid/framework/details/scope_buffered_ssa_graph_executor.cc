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

#include "paddle/fluid/framework/details/scope_buffered_ssa_graph_executor.h"
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {
ScopeBufferedSSAGraphExecutor::ScopeBufferedSSAGraphExecutor(
    ExecutionStrategy strategy, std::vector<Scope *> local_scopes,
    std::vector<VariableInfo> var_infos, std::vector<platform::Place> places,
    std::unique_ptr<SSAGraphExecutor> &&underlying_executor)
    : strategy_(std::move(strategy)),
      underlying_executor_(std::move(underlying_executor)),
      local_scopes_(std::move(local_scopes)),
      var_infos_(std::move(var_infos)),
      places_(std::move(places)) {}

FeedFetchList ScopeBufferedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  if (init_variable) {
    platform::RecordEvent e("InitLocalExeScopes");
    PrepareLocalExeScopes();
    init_variable = false;
  }

  std::vector<framework::LoDTensor> fetch_data;
  std::exception_ptr eptr = nullptr;
  try {
    fetch_data = underlying_executor_->Run(fetch_tensors);
  } catch (...) {
    eptr = std::current_exception();
  }

  ++drop_scope_counter_;
  if (drop_scope_counter_ == strategy_.num_iteration_per_drop_scope_) {
    CleanLocalExeScopes();
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  } else {
    return fetch_data;
  }
}

void ScopeBufferedSSAGraphExecutor::CleanLocalExeScopes() {
  platform::RecordEvent drop_scope_event("CleanLocalExeScopes");
  drop_scope_counter_ = 0;
  SyncDevices();

  for (size_t scope_idx = 0; scope_idx < local_scopes_.size(); ++scope_idx) {
    auto &scope = local_scopes_.at(scope_idx);
    auto &local_scope =
        *scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>();
    local_scope->DropKids();

    {
      auto local_var_names = local_scope->LocalVarNames();
      std::set<std::string> local_var_set(local_var_names.begin(),
                                          local_var_names.end());
      std::vector<std::string> result;
      auto &origin_local_vars = local_scopes_var_name_.at(scope_idx);
      result.reserve(origin_local_vars.size());
      std::set_intersection(local_var_set.begin(), local_var_set.end(),
                            origin_local_vars.begin(), origin_local_vars.end(),
                            std::back_inserter(result));
      // the intersection of the two set should be origin_local_vars.
      PADDLE_ENFORCE_EQ(result.size(), origin_local_vars.size());
      result.clear();
      std::set_difference(local_var_set.begin(), local_var_set.end(),
                          origin_local_vars.begin(), origin_local_vars.end(),
                          std::back_inserter(result));
      local_scope->EraseVars(result);
    }
    VLOG(3) << "Clean local execution scope: " << local_scope;
  }
}

void ScopeBufferedSSAGraphExecutor::DropLocalExeScopes() {
  platform::RecordEvent drop_scope_event("DropLocalExeScopes");
  drop_scope_counter_ = 0;
  init_variable = true;
  SyncDevices();

  for (auto &scope : local_scopes_) {
    auto *local_scope_var = scope->FindLocalVar(details::kLocalExecScopeName);
    if (local_scope_var != nullptr) {
      auto &local_scope = *local_scope_var->GetMutable<Scope *>();
      scope->DeleteScope(local_scope);
      VLOG(3) << "Drop local execution scope: " << local_scope;
    }
  }
}

void ScopeBufferedSSAGraphExecutor::SyncDevices() const {
  for (auto p : places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }
}

void ScopeBufferedSSAGraphExecutor::PrepareLocalExeScopes() {
  // Create local scopes.
  local_scopes_var_name_.clear();
  local_scopes_var_name_.resize(local_scopes_.size());
  size_t idx = local_scopes_.size() - 1;
  for (auto it = local_scopes_.rbegin(); it != local_scopes_.rend();
       ++it, --idx) {
    auto &scope = *it;
    Scope &local_scope = scope->NewScope();
    *scope->Var(kLocalExecScopeName)->GetMutable<Scope *>() = &local_scope;

    for (auto &info : var_infos_) {
      if (scope->FindVar(info.name_) != nullptr) {
        continue;
      }
      if (info.persistable_) {  // Persistable
        InitializeVariable(scope->Var(info.name_), info.type_);
      } else {
        InitializeVariable(local_scope.Var(info.name_), info.type_);
      }
    }
    auto local_var_names = local_scope.LocalVarNames();
    local_scopes_var_name_.at(idx).insert(local_var_names.begin(),
                                          local_var_names.end());
  }
}

bool ScopeBufferedSSAGraphExecutor::NeedCreateLocalExeScope() {
  return drop_scope_counter_ == 0;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
