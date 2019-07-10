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
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/op_registry.h"
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
  pre_local_exec_scopes_.resize(local_exec_scopes_.size());
  post_local_exec_scopes_.resize(local_exec_scopes_.size());
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

  RecordHistoryLocalExecScopes(fetch_tensors.size() > 0, [&]() {
    try {
      fetch_data = underlying_executor_->Run(fetch_tensors);
    } catch (...) {
      eptr = std::current_exception();
    }
  });

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

void ScopeBufferedSSAGraphExecutor::RecordHistoryLocalExecScopes(
    bool has_fetch, const std::function<void()> &callback) {
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    pre_local_exec_scopes_.at(scope_id).clear();
    auto scopes = local_exec_scopes_.at(scope_id)->kids();
    VLOG(10) << "pre_local_exec_scopes[" << scope_id
             << "] sub-scope: " << scopes.size();
    pre_local_exec_scopes_.at(scope_id).insert(scopes.begin(), scopes.end());
  }

  callback();

  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    post_local_exec_scopes_.at(scope_id).clear();
    auto scopes = local_exec_scopes_.at(scope_id)->kids();
    VLOG(10) << "post_local_exec_scopes[" << scope_id
             << "] sub-scope: " << scopes.size();
    post_local_exec_scopes_.at(scope_id).insert(scopes.begin(), scopes.end());
  }

  history_local_exec_scopes_.emplace_back();
  auto &incr_local_exec_scopes = history_local_exec_scopes_.back();
  incr_local_exec_scopes.resize(local_exec_scopes_.size());
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    for (auto &scope : post_local_exec_scopes_.at(scope_id)) {
      if (!pre_local_exec_scopes_.at(scope_id).count(scope)) {
        incr_local_exec_scopes.at(scope_id).insert(scope);
      }
    }
    if (VLOG_IS_ON(10)) {
      std::stringstream out;
      out << scope_id << " kids: ";
      for (auto &scope : incr_local_exec_scopes.at(scope_id)) {
        out << scope << ", ";
      }
      VLOG(10) << out.str();
    }
  }

  size_t history_step = history_local_exec_scopes_.size();
  if (has_fetch && history_step >= 2) {
    VLOG(10) << "delete pre_incr_local_exec_scopes.";
    for (size_t i = 0; i < history_step - 1; ++i) {
      auto &pre_incr_local_exec_scopes = history_local_exec_scopes_.front();
      for (size_t scope_idx = 0; scope_idx < pre_incr_local_exec_scopes.size();
           ++scope_idx) {
        for (auto scope : pre_incr_local_exec_scopes[scope_idx]) {
          local_exec_scopes_.at(scope_idx)->DeleteScope(scope);
        }
      }
      history_local_exec_scopes_.pop_front();
    }
  }
}

void ScopeBufferedSSAGraphExecutor::InitVariables() {
  for (auto &info : tmp_var_infos_) {
    for (auto &pair : info) {
      InitializeVariable(pair.first, pair.second);
    }
  }

  const ir::Graph &graph = Graph();
  if (graph.Has(details::kProgramDescs)) {
    auto &program_descs =
        graph.Get<details::ProgramDescs>(details::kProgramDescs);
    // Init vars
    auto &fused_grad_vars = graph.Get<details::FusedVars>(details::kFusedVars);
    for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
      for (auto &var_name : fused_grad_vars) {
        auto var = local_exec_scopes_[i]->Var(var_name);
        var->GetMutable<LoDTensor>();
      }
    }

    for (auto &program_desc : program_descs) {
      for (auto &op_desc : program_desc.Block(0).AllOps()) {
        for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
          auto op = OpRegistry::CreateOp(*op_desc);
          op->Run(*local_exec_scopes_[i], places_[i]);
        }
      }
    }
  }
}

void ScopeBufferedSSAGraphExecutor::DropLocalExeScopes() {
  platform::RecordEvent drop_scope_event("DropLocalExeScopes");
  drop_scope_counter_ = 0;
  for (auto &p : places_) {
    platform::DeviceContextPool::Instance().Get(p)->Wait();
  }

  history_local_exec_scopes_.clear();
  for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
    local_exec_scopes_[i]->EraseVarsExcept(preserve_vars_[i]);
    local_exec_scopes_[i]->DropKids();
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
      if (info.persistable_) {  // Persistable
        auto var = scope->FindVar(info.name_);
        if (var != nullptr) {
          VLOG(2)
              << info.name_
              << " has been initialized beforehand in global scope, skipped";
          continue;
        }
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
