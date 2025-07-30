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
#include "paddle/fluid/framework/scope_pool.h"

namespace paddle::framework {

ScopePool &ScopePool::Instance() {  // NOLINT
  static ScopePool pool;
  return pool;
}

void ScopePool::DeleteScope(Scope *scope) { delete scope; }

void ScopePool::Insert(std::unique_ptr<Scope> &&s) {
  std::lock_guard<std::mutex> guard(mtx_);
  scopes_.insert(s.release());
}

void ScopePool::Remove(Scope *s) {
  size_t has_scope = 0;
  {
    std::lock_guard<std::mutex> guard(mtx_);
    has_scope = scopes_.erase(s);
  }
  PADDLE_ENFORCE_GT(
      has_scope,
      0,
      common::errors::NotFound("Global scope %p is not found in ScopePool. "
                               "Deleting a nonexistent scope is not allowed.",
                               s));
  DeleteScope(s);
}

ScopePool::~ScopePool() { Clear(); }

void ScopePool::Clear() {
  std::lock_guard<std::mutex> guard(mtx_);
  for (auto *s : scopes_) {
    DeleteScope(s);
  }
  scopes_.clear();
}

}  // namespace paddle::framework
