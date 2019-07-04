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
    std::vector<Scope *> local_exec_scopes, std::vector<VariableInfo> var_infos,
    std::vector<platform::Place> places,
    std::unique_ptr<SSAGraphExecutor> &&underlying_executor)
    : strategy_(std::move(strategy)),
      underlying_executor_(std::move(underlying_executor)),
      local_scopes_(std::move(local_scopes)),
      local_exec_scopes_(std::move(local_exec_scopes)),
      var_infos_(std::move(var_infos)),
      places_(std::move(places)) {
  PADDLE_ENFORCE_EQ(local_scopes_.size(), local_exec_scopes_.size());
  PrepareLocalExeScopes();
}

FeedFetchList ScopeBufferedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  if (drop_scope_counter_ == 0) {
    platform::RecordEvent e("InitLocalVars");
    InitVariables();
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
    DropLocalExeScopes();
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  } else {
    return fetch_data;
  }
}

void ScopeBufferedSSAGraphExecutor::InitVariables() {
  for (auto &info : tmp_var_infos_) {
    for (auto &pair : info) {
      InitializeVariable(pair.first, pair.second);
    }
  }
}

void ScopeBufferedSSAGraphExecutor::DropLocalExeScopes() {
  platform::RecordEvent drop_scope_event("DropLocalExeScopes");
  drop_scope_counter_ = 0;
  for (auto &p : places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }

  for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
    local_exec_scopes_[i]->ClearWithPreserve(preserve_vars_[i]);
    for (auto &preserve_var : preserve_vars_[i]) {
      preserve_var->Clear();
    }
    VLOG(3) << "Drop local execution scope: " << local_scopes_[i];
  }
}

void ScopeBufferedSSAGraphExecutor::PrepareLocalExeScopes() {
  // Create local scopes.
  preserve_vars_.resize(local_scopes_.size());
  tmp_var_infos_.resize(local_scopes_.size());

  for (auto it = local_scopes_.rbegin(); it != local_scopes_.rend(); ++it) {
    size_t idx = local_scopes_.size() - 1 - (it - local_scopes_.rbegin());
    auto *scope = local_scopes_[idx];
    auto *local_scope = local_exec_scopes_[idx];

    for (auto &info : var_infos_) {
      if (scope->FindVar(info.name_) != nullptr) {
        if (!info.persistable_) {
          VLOG(2) << info.name_ << " is found in parent scope, skipped";
        }
        continue;
      }

      if (info.persistable_) {  // Persistable
        InitializeVariable(scope->Var(info.name_), info.type_);
      } else {
        Variable *tmp_var = local_scope->Var(info.name_);
        preserve_vars_[idx].emplace(tmp_var);
        tmp_var_infos_[idx].emplace_back(tmp_var, info.type_);
      }
    }
  }
}

bool ScopeBufferedSSAGraphExecutor::NeedCreateLocalExeScope() {
  return drop_scope_counter_ == 0;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
