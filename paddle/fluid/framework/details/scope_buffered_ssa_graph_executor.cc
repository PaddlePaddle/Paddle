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
    std::vector<VariableInfo> var_infos, std::vector<platform::Place> places,
    std::unique_ptr<SSAGraphExecutor> &&underlying_executor)
    : strategy_(std::move(strategy)),
      underlying_executor_(std::move(underlying_executor)),
      local_scopes_(std::move(local_scopes)),
      var_infos_(std::move(var_infos)),
      places_(std::move(places)) {
  sub_local_scopes_.reserve(local_scopes_.size());
  non_persistable_var_names_.resize(local_scopes_.size());
  for (auto &local_scope : local_scopes_) {
    auto &new_scope = local_scope->NewScope();
    *local_scope->Var(details::kLocalExecScopeName)->GetMutable<Scope *>() =
        &new_scope;
    sub_local_scopes_.emplace_back(&new_scope);
  }
}

FeedFetchList ScopeBufferedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  if (drop_scope_counter_ == 0) {
    for (size_t i = 0; i < local_scopes_.size(); ++i) {
      for (auto &info : var_infos_) {
        if (local_scopes_[i]->FindVar(info.name_) != nullptr) {
          continue;
        }

        if (info.persistable_) {  // Persistable
          LOG(WARNING) << "Variable " << info.name_
                       << " is persistable but not initialized";
          InitializeVariable(local_scopes_[i]->Var(info.name_), info.type_);
        } else {
          non_persistable_var_names_[i].emplace(info.name_);
          InitializeVariable(sub_local_scopes_[i]->Var(info.name_), info.type_);
        }
      }
    }
  }

  std::vector<framework::LoDTensor> fetch_data;
  std::exception_ptr eptr = nullptr;
  try {
    fetch_data = underlying_executor_->Run(fetch_tensors);
  } catch (...) {
    eptr = std::current_exception();
  }

  platform::RecordEvent e("ScopeBufferedSSAGraphExecutorAfterRun");
  ++drop_scope_counter_;

  if (drop_scope_counter_ == strategy_.num_iteration_per_drop_scope_) {
    WaitComputationalStreams();

    for (size_t i = 0; i < sub_local_scopes_.size(); ++i) {
      sub_local_scopes_[i]->ClearVarsAndDropOthers(
          non_persistable_var_names_[i]);
      non_persistable_var_names_[i].clear();
    }

    drop_scope_counter_ = 0;
  }

  if (eptr) {
    std::rethrow_exception(eptr);
  } else {
    return fetch_data;
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
