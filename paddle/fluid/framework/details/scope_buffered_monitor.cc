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

#include "paddle/fluid/framework/details/scope_buffered_monitor.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

DECLARE_double(local_exe_sub_scope_limit);

namespace paddle {
namespace framework {
namespace details {

static constexpr double kMB = 1 / (1024 * 1024);

static void GetTensors(Variable *var,
                       std::unordered_set<Tensor *> *tensor_set) {
  if (var->IsType<LoDTensor>() && var->Get<LoDTensor>().IsInitialized()) {
    tensor_set->insert(var->GetMutable<LoDTensor>());
  } else if (var->IsType<pten::SelectedRows>() &&
             var->Get<pten::SelectedRows>().value().IsInitialized()) {
    tensor_set->insert(var->GetMutable<pten::SelectedRows>()->mutable_value());
  } else if (var->IsType<LoDTensorArray>()) {
    auto *tensor_arr = var->GetMutable<LoDTensorArray>();
    for (auto &t : *tensor_arr) {
      if (t.IsInitialized()) {
        tensor_set->insert(&t);
      }
    }
  }
}

static void GetTensors(Scope *scope, std::unordered_set<Tensor *> *tensor_set) {
  for (auto &var_name : scope->LocalVarNames()) {
    GetTensors(scope->FindVar(var_name), tensor_set);
  }

  for (auto *kid : scope->kids()) {
    GetTensors(kid, tensor_set);
  }
}

static size_t GetTensorMemorySize(Scope *scope, bool clear_cpu_tensor) {
  std::unordered_set<Tensor *> tensor_set;
  GetTensors(scope, &tensor_set);
  size_t memory_size = 0;
  std::unordered_set<memory::Allocation *> allocation_set;
  for (auto *tensor : tensor_set) {
    if (clear_cpu_tensor && platform::is_cpu_place(tensor->place())) {
      tensor->clear();
    } else {
      auto allocation = tensor->Holder().get();
      if (!allocation_set.count(allocation)) {
        memory_size += allocation->size();
        allocation_set.insert(allocation);
      }
    }
  }
  return memory_size;
}

size_t GetScopeVarMemorySize(Scope *scope) {
  return GetTensorMemorySize(scope, false /*clear_cpu_tensor*/);
}

ScopeBufferedMonitor::ScopeBufferedMonitor(
    const std::vector<platform::Place> &places,
    const std::vector<Scope *> &local_exec_scopes)
    : places_(places), local_exec_scopes_(local_exec_scopes) {
  pre_local_exec_scopes_.resize(local_exec_scopes_.size());
  post_local_exec_scopes_.resize(local_exec_scopes_.size());
}

void ScopeBufferedMonitor::Apply(const std::function<void()> &callback,
                                 bool has_fetch) {
  std::unique_ptr<platform::RecordEvent> pre_local_exec_scopes_event(
      new platform::RecordEvent(
          "ScopeBufferedMonitor::pre_local_exec_scopes_process"));
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    pre_local_exec_scopes_.at(scope_id).clear();
    auto scopes = local_exec_scopes_.at(scope_id)->kids();
    VLOG(10) << "pre_local_exec_scopes[" << scope_id
             << "] sub-scope: " << scopes.size();
    pre_local_exec_scopes_.at(scope_id).insert(scopes.begin(), scopes.end());
  }
  pre_local_exec_scopes_event.reset();

  callback();

  std::unique_ptr<platform::RecordEvent> post_local_exec_scopes_event(
      new platform::RecordEvent(
          "ScopeBufferedMonitor::post_local_exec_scopes_process"));
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
      if (incr_local_exec_scopes.at(scope_id).size() &&
          FLAGS_local_exe_sub_scope_limit > 0) {
        VLOG(10)
            << "FLAGS_local_exe_sub_scope_limit is "
            << FLAGS_local_exe_sub_scope_limit
            << " MBytes now. If you don't need to limit the memory of local "
               "execution scope, you should set "
               "FLAGS_local_exe_sub_scope_limit=-1.";
      }
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
    ClearHistoryLocalExecScopes(history_step - 1);
  }

  // Delete CPU Memory
  std::vector<size_t> gpu_memory_size_per_gpu(places_.size());
  for (auto &scope_vec : history_local_exec_scopes_) {
    for (size_t idx = 0; idx < scope_vec.size(); ++idx) {
      for (auto &scope : scope_vec.at(idx)) {
        gpu_memory_size_per_gpu.at(idx) +=
            GetTensorMemorySize(scope, true /*clear_cpu_tensor*/);
      }
    }
  }
  if (VLOG_IS_ON(8)) {
    for (size_t idx = 0; idx < gpu_memory_size_per_gpu.size(); ++idx) {
      VLOG(8) << "history local exec scopes contains "
              << string::HumanReadableSize(gpu_memory_size_per_gpu.at(idx))
              << " in " << places_.at(idx);
    }
  }

  if (FLAGS_local_exe_sub_scope_limit > 0) {
    for (size_t idx = 0; idx < gpu_memory_size_per_gpu.size(); ++idx) {
      if (gpu_memory_size_per_gpu.at(idx) / kMB >=
          FLAGS_local_exe_sub_scope_limit) {
        platform::DeviceContextPool::Instance().Get(places_.at(idx))->Wait();
        local_exec_scopes_.at(idx)->DropKids();
      }
      for (auto &scope_vec : history_local_exec_scopes_) {
        scope_vec.at(idx).clear();
      }
    }
  }
}

void ScopeBufferedMonitor::ClearHistoryLocalExecScopes(size_t history_step) {
  VLOG(10) << "delete pre_incr_local_exec_scopes.";
  for (size_t i = 0; i < history_step; ++i) {
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

void ScopeBufferedMonitor::ClearHistoryLocalExecScopes() {
  history_local_exec_scopes_.clear();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
