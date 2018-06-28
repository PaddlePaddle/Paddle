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

#include "paddle/fluid/operators/distributed/brpc_rdma_pool.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

RdmaPool& RdmaPool::Instance() {
  static RdmaPool* g_rdma_mem_pool = new RdmaPool();
  return *g_rdma_mem_pool;
}

void* RdmaPool::VarMem(const std::string& varname, int64_t size) {
  auto* info = Get(varname);
  if (nullptr == info) {
    platform::CUDAPinnedPlace cuda_pinned;
    void* data = memory::Alloc(cuda_pinned, size);
    Insert(varname, data, size);
    return data;
  }

  if (info->data_size != size) {
    PADDLE_ENFORCE(false, "var size:%ld != %ld", size, info->data_size);
    return nullptr;
  }

  return info->data;
}

void RdmaPool::Insert(const std::string& varname, void* data,
                      int64_t data_size) {
  boost::unique_lock<boost::shared_mutex> m(access_);
  auto it = pool_.find(varname);
  if (it != pool_.end()) {
    PADDLE_ENFORCE(false, "%s has registered!", varname);
    return;
  }

  VarInfo info;
  info.data = data;
  info.data_size = data_size;
  pool_[varname] = info;
  return;
}

RdmaPool::VarInfo* RdmaPool::Get(const std::string& varname) {
  boost::shared_lock<boost::shared_mutex> m(access_);
  auto it = pool_.find(varname);
  if (it == pool_.end()) {
    return nullptr;
  }

  return &it->second;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
