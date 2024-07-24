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

#include "paddle/fluid/framework/transfer_scope_cache.h"

namespace paddle::framework {

std::unordered_map<size_t, Scope*>& global_transfer_data_cache() {
  thread_local auto* x = new std::unordered_map<size_t, Scope*>;
  return *x;
}

std::unordered_set<Scope*>& global_transfer_scope_cache() {
  thread_local auto* x = new std::unordered_set<Scope*>;
  return *x;
}

std::unordered_map<const Scope*, std::unordered_set<size_t>>&
global_transfer_scope_key() {
  thread_local auto* x =
      new std::unordered_map<const Scope*, std::unordered_set<size_t>>;
  return *x;
}

Scope* TryCreateTransferScope(const phi::KernelKey& type0,
                              const phi::KernelKey& type1,
                              const Scope* scope) {
  Scope* new_scope{nullptr};
  size_t infer_cache_key =
      CombineHash(static_cast<size_t>(phi::KernelKey::Hash()(type0)),
                  static_cast<size_t>(phi::KernelKey::Hash()(type1)));
  infer_cache_key =
      CombineHash(infer_cache_key, std::hash<const Scope*>()(scope));

  global_transfer_scope_key()[scope].insert(infer_cache_key);  //  NOLINT

  auto it = global_transfer_data_cache().find(infer_cache_key);
  if (it != global_transfer_data_cache().end()) {
    new_scope = global_transfer_data_cache()[infer_cache_key];
  } else {
    new_scope = &scope->NewScope();
    global_transfer_data_cache()[infer_cache_key] = new_scope;
  }
  global_transfer_scope_cache().insert(new_scope);
  return new_scope;
}

}  // namespace paddle::framework
