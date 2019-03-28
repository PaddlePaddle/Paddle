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

#pragma once
#ifdef PADDLE_WITH_BRPC_RDMA

#include <pthread.h>  // NOLINT
#include <string>
#include <unordered_map>

namespace paddle {
namespace operators {
namespace distributed {

/*
 * This class is used to avoid duplicated registion of brpc::rdma.
 */
class RdmaMemPool {
 public:
  static RdmaMemPool& Instance();
  RdmaMemPool() : access_(PTHREAD_RWLOCK_INITIALIZER) {}

  virtual ~RdmaMemPool() { pthread_rwlock_destroy(&access_); }

  void Register(const std::string& varname, void* data, int64_t size);
  void* Find(const std::string& varname, int64_t size);

 private:
  struct VarInfo {
    void* data;
    int64_t data_size;

    VarInfo() : data(nullptr), data_size(0) {}
  };

 private:
  std::unordered_map<std::string, VarInfo> pool_;
  pthread_rwlock_t access_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

#endif
