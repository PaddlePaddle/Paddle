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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/profiler.h"
namespace paddle {
namespace framework {
namespace details {

ScopeBufferedMonitor::ScopeBufferedMonitor(
    const std::vector<Scope *> &local_exec_scopes)
    : local_exec_scopes_(local_exec_scopes) {
  pre_local_exec_scopes_.resize(local_exec_scopes_.size());
  post_local_exec_scopes_.resize(local_exec_scopes_.size());
}

void ScopeBufferedMonitor::Run(const std::function<void()> &callback,
                               bool has_fetch) {
  std::unique_ptr<platform::RecordEvent> pre_local_exec_scopes_event(
      new platform::RecordEvent(
          "ScopeBufferedMonitor::pre_local_exec_scopes process."));
  for (size_t scope_id = 0; scope_id < local_exec_scopes_.size(); ++scope_id) {
    pre_local_exec_scopes_.at(scope_id).clear();
    auto scopes = local_exec_scopes_.at(scope_id)->kids();
    VLOG(10) << "pre_local_exec_scopes[" << scope_id
             << "] sub-scope: " << scopes.size();
    pre_local_exec_scopes_.at(scope_id).insert(scopes.begin(), scopes.end());
  }
  pre_local_exec_scopes_event.release();

  callback();

  std::unique_ptr<platform::RecordEvent> post_local_exec_scopes_event(
      new platform::RecordEvent(
          "ScopeBufferedMonitor::post_local_exec_scopes process."));
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

void ScopeBufferedMonitor::ClearHistoryLocalExecScopes() {
  history_local_exec_scopes_.clear();
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
