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
#include <string>
#include <thread>
#include <unordered_map>
#ifdef _POSIX_C_SOURCE
#include <time.h>
#endif
#include "paddle/fluid/platform/macros.h"  // import DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace internal {
static uint64_t main_tid =
    std::hash<std::thread::id>()(std::this_thread::get_id());

template <typename T>
class ThreadDataRegistry {
  class ThreadDataHolder;

 public:
  // Singleton
  static ThreadDataRegistry& GetInstance() {
    static ThreadDataRegistry instance;
    return instance;
  }

  const T& GetCurrentThreadData() { return CurrentThreadData(); }

  void SetCurrentThreadData(const T& val) {
    std::lock_guard<std::mutex> lock(lock_);
    CurrentThreadData() = val;
  }

  // Returns current snapshot of all threads. Make sure there is no thread
  // create/destory when using it.
  template <typename = std::enable_if_t<std::is_copy_constructible<T>::value>>
  std::unordered_map<uint64_t, T> GetAllThreadDataByValue() {
    std::unordered_map<uint64_t, T> data_copy;
    std::lock_guard<std::mutex> lock(lock_);
    data_copy.reserve(tid_map_.size());
    for (auto& kv : tid_map_) {
      data_copy.emplace(kv.first, kv.second->GetData());
    }
    return std::move(data_copy);
  }

  void RegisterData(uint64_t tid, ThreadDataHolder* tls_obj) {
    std::lock_guard<std::mutex> lock(lock_);
    tid_map_[tid] = tls_obj;
  }

  void UnregisterData(uint64_t tid) {
    if (tid == main_tid) {
      return;
    }
    std::lock_guard<std::mutex> lock(lock_);
    tid_map_.erase(tid);
  }

 private:
  class ThreadDataHolder {
   public:
    ThreadDataHolder() {
      tid_ = std::hash<std::thread::id>()(std::this_thread::get_id());
      ThreadDataRegistry::GetInstance().RegisterData(tid_, this);
    }

    ~ThreadDataHolder() {
      ThreadDataRegistry::GetInstance().UnregisterData(tid_);
    }

    T& GetData() { return data_; }

   private:
    uint64_t tid_;
    T data_;
  };

  ThreadDataRegistry() = default;

  DISABLE_COPY_AND_ASSIGN(ThreadDataRegistry);

  T& CurrentThreadData() {
    static thread_local ThreadDataHolder thread_data;
    return thread_data.GetData();
  }

  std::mutex lock_;
  std::unordered_map<uint64_t, ThreadDataHolder*> tid_map_;  // not owned
};

}  // namespace internal

// Get system-wide realtime clock in nanoseconds
inline uint64_t PosixInNsec() {
#ifdef _POSIX_C_SOURCE
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec * 1000 * 1000 * 1000 + tp.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return 1000 * (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
#endif
}

// All kinds of Ids for OS thread
struct ThreadId {
  uint64_t std_tid = 0;    // std::hash<std::thread::id>
  uint64_t sys_tid = 0;    // OS-specific, Linux: gettid
  uint32_t cupti_tid = 0;  // thread_id used by Nvidia CUPTI
};

// Better performance than GetCurrentThreadId
uint64_t GetCurrentThreadStdId();

// Better performance than GetCurrentThreadId
uint64_t GetCurrentThreadSysId();

ThreadId GetCurrentThreadId();

// Return the map from StdTid to ThreadId
// Returns current snapshot of all threads. Make sure there is no thread
// create/destory when using it.
std::unordered_map<uint64_t, ThreadId> GetAllThreadIds();

// Returns 'unset' if SetCurrentThreadName is never called.
std::string GetCurrentThreadName();

// Return the map from StdTid to ThreadName
// Returns current snapshot of all threads. Make sure there is no thread
// create/destory when using it.
std::unordered_map<uint64_t, std::string> GetAllThreadNames();

// Thread name is immutable, only the first call will succeed.
// Returns false on failure.
bool SetCurrentThreadName(const std::string& name);

uint32_t GetProcessId();

}  // namespace platform
}  // namespace paddle
