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

#ifdef PADDLE_WITH_MKLDNN
using transfer_data_cache_map = std::unordered_map<size_t, Scope*>;
using transfer_scope_cache_map = std::unordered_set<Scope*>;
static std::unordered_map<size_t, transfer_data_cache_map*>
    static_transfer_data_caches;
static std::unordered_map<size_t, transfer_scope_cache_map*>
    static_transfer_scope_caches;
#endif

std::unordered_map<size_t, Scope*>& global_transfer_data_cache() {
#ifdef PADDLE_WITH_MKLDNN
  size_t sid = platform::get_cur_mkldnn_session_id();

  // if there is specific mkldnn tid setting from user.
  if (sid != platform::kMKLDNNSessionID_Default) {
    sid = std::hash<std::thread::id>()(std::this_thread::get_id());

    static std::mutex acquire_barrier;
    std::lock_guard<std::mutex> block_until_finish_this_job(acquire_barrier);

    auto map_it = static_transfer_data_caches.find(sid);
    if (map_it == static_transfer_data_caches.end()) {
      auto* x = new transfer_data_cache_map;
      static_transfer_data_caches[sid] = x;
      return *x;
    } else {
      return *static_transfer_data_caches[sid];
    }
  }
#endif
  thread_local auto* x = new std::unordered_map<size_t, Scope*>;
  return *x;
}

std::unordered_set<Scope*>& global_transfer_scope_cache() {
#ifdef PADDLE_WITH_MKLDNN
  size_t sid = platform::get_cur_mkldnn_session_id();

  // if there is specific mkldnn session id setting from user.
  if (sid != platform::kMKLDNNSessionID_Default) {
    sid = std::hash<std::thread::id>()(std::this_thread::get_id());

    static std::mutex acquire_barrier;
    std::lock_guard<std::mutex> block_until_finish_this_job(acquire_barrier);

    auto map_it = static_transfer_scope_caches.find(sid);
    if (map_it == static_transfer_scope_caches.end()) {
      auto* x = new transfer_scope_cache_map;
      static_transfer_scope_caches[sid] = x;
      return *x;
    } else {
      return *static_transfer_scope_caches[sid];
    }
  }
#endif
  thread_local auto* x = new std::unordered_set<Scope*>;
  return *x;
}

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
