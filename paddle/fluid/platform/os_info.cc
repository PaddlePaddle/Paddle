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

#include "paddle/fluid/platform/os_info.h"
#include <functional>
#include <sstream>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(_MSC_VER)
#include <processthreadsapi.h>
#else
#include <unistd.h>
#endif
#include "paddle/fluid/framework/new_executor/workqueue/thread_data_registry.h"
#include "paddle/fluid/platform/macros.h"  // import DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace platform {
namespace internal {

using framework::ThreadDataRegistry;

class InternalThreadId {
 public:
  InternalThreadId();

  const ThreadId& GetTid() const { return id_; }

 private:
  ThreadId id_;
};

InternalThreadId::InternalThreadId() {
  // C++ std tid
  id_.std_tid = std::hash<std::thread::id>()(std::this_thread::get_id());
// system tid
#if defined(__linux__)
  id_.sys_tid = static_cast<uint64_t>(syscall(SYS_gettid));
#elif defined(_MSC_VER)
  id_.sys_tid = static_cast<uint64_t>(::GetCurrentThreadId());
#else  // unsupported platforms, use std_tid
  id_.sys_tid = id_.std_tid;
#endif
  // cupti tid
  std::stringstream ss;
  ss << std::this_thread::get_id();
  id_.cupti_tid = static_cast<uint32_t>(std::stoull(ss.str()));
}

}  // namespace internal

uint64_t GetCurrentThreadSysId() {
  return internal::ThreadDataRegistry<internal::InternalThreadId>::GetInstance()
      .GetCurrentThreadData()
      .GetTid()
      .sys_tid;
}

uint64_t GetCurrentThreadStdId() {
  return internal::ThreadDataRegistry<internal::InternalThreadId>::GetInstance()
      .GetCurrentThreadData()
      .GetTid()
      .std_tid;
}

ThreadId GetCurrentThreadId() {
  return internal::ThreadDataRegistry<internal::InternalThreadId>::GetInstance()
      .GetCurrentThreadData()
      .GetTid();
}

std::unordered_map<uint64_t, ThreadId> GetAllThreadIds() {
  auto tids =
      internal::ThreadDataRegistry<internal::InternalThreadId>::GetInstance()
          .GetAllThreadDataByValue();
  std::unordered_map<uint64_t, ThreadId> res;
  for (const auto& kv : tids) {
    res[kv.first] = kv.second.GetTid();
  }
  return res;
}

std::string GetCurrentThreadName() {
  const auto& thread_name =
      internal::ThreadDataRegistry<std::string>::GetInstance()
          .GetCurrentThreadData();
  return thread_name.empty() ? kDefaultThreadName : thread_name;
}

std::unordered_map<uint64_t, std::string> GetAllThreadNames() {
  return internal::ThreadDataRegistry<std::string>::GetInstance()
      .GetAllThreadDataByValue();
}

bool SetCurrentThreadName(const std::string& name) {
  auto& instance = internal::ThreadDataRegistry<std::string>::GetInstance();
  const auto& cur_name = instance.GetCurrentThreadData();
  if (!cur_name.empty() || name.empty() || name == kDefaultThreadName) {
    return false;
  }
  instance.SetCurrentThreadData(name);
  return true;
}

uint32_t GetProcessId() {
#if defined(_MSC_VER)
  return static_cast<uint32_t>(GetCurrentProcessId());
#else
  return static_cast<uint32_t>(getpid());
#endif
}

}  // namespace platform
}  // namespace paddle
