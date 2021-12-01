/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {

// All kinds of Ids for OS thread
class ThreadId {
 public:
  ThreadId();

  uint64_t StdTid() const { return std_tid_; }

  uint32_t CuptiTid() const { return cupti_tid_; }

  uint64_t SysTid() const { return sys_tid_; }

 private:
  uint64_t std_tid_ = 0;    // main id, std::hash<std::thread::id>
  uint32_t cupti_tid_ = 0;  // thread_id used by Nvidia CUPTI
  uint64_t sys_tid_ = 0;    // OS-specific, Linux: gettid
};

class ThreadIdRegistry {
 public:
  // singleton
  static ThreadIdRegistry& GetInstance() {
    static ThreadIdRegistry instance;
    return instance;
  }

  const ThreadId* GetThreadId(uint64_t std_id) {
    std::lock_guard<std::mutex> lock(lock_);
    if (LIKELY(id_map_.find(std_id) != id_map_.end())) {
      return id_map_[std_id];
    }
    return nullptr;
  }

  const ThreadId& CurrentThreadId() {
    static thread_local ThreadId* tid_ = nullptr;
    if (LIKELY(tid_ != nullptr)) {
      return *tid_;
    }
    tid_ = new ThreadId;
    std::lock_guard<std::mutex> lock(lock_);
    id_map_[tid_->StdTid()] = tid_;
    return *tid_;
  }

 private:
  ThreadIdRegistry() = default;
  DISABLE_COPY_AND_ASSIGN(ThreadIdRegistry);
  ~ThreadIdRegistry();

  std::mutex lock_;
  std::unordered_map<uint64_t, ThreadId*> id_map_;
};

}  // namespace platform
}  // namespace paddle
