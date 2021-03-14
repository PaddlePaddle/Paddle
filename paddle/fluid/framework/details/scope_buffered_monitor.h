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

#pragma once
#include <deque>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace details {

class ScopeBufferedMonitor {
 public:
  ScopeBufferedMonitor(const std::vector<platform::Place> &places,
                       const std::vector<Scope *> &local_exec_scopes);

  void Apply(const std::function<void()> &callback, bool has_fetch);

  void ClearHistoryLocalExecScopes();

  void ClearHistoryLocalExecScopes(size_t history_step);

 private:
  std::vector<platform::Place> places_;
  std::vector<Scope *> local_exec_scopes_;
  std::vector<std::unordered_set<Scope *>> pre_local_exec_scopes_;
  std::vector<std::unordered_set<Scope *>> post_local_exec_scopes_;
  std::deque<std::vector<std::unordered_set<Scope *>>>
      history_local_exec_scopes_;
};

size_t GetScopeVarMemorySize(Scope *scope);

}  // namespace details
}  // namespace framework
}  // namespace paddle
