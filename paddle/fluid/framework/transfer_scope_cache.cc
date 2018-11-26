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

namespace paddle {
namespace framework {

// Holds all the transfer scope across the process.
std::unordered_map<size_t, Scope*>& global_transfer_data_cache() {
  typedef std::unordered_map<size_t, Scope*> map_t;
  thread_local std::unique_ptr<map_t> x(new map_t);
  return *x;
}

// Holds all the transfer scope for this thread.
std::unordered_set<Scope*>& global_transfer_scope_cache() {
  typedef std::unordered_set<Scope*> set_t;
  thread_local std::unique_ptr<set_t> x(new set_t);
  return *x;
}

// Try to create a transfer scope. If one cached scope has match the
// requirement, just return that one.
// Inputs:
// @type0: the source kernel type.
// @type1: the target kernel type.
// @scope: the execution scope of this op.
// Returns: A scope used to hold the transfer data across the different kernel
// type.
Scope* TryCreateTransferScope(OpKernelType type0, OpKernelType type1,
                              const Scope* scope) {
  Scope* new_scope{nullptr};
  size_t infer_cache_key =
      CombineHash(OpKernelType::Hash()(type0), OpKernelType::Hash()(type1));
  infer_cache_key =
      CombineHash(infer_cache_key, std::hash<const Scope*>()(scope));

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

}  // namespace framework
}  // namespace paddle
