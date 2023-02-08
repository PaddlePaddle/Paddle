/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <unordered_map>
#ifdef _POSIX_C_SOURCE
#include <time.h>
#endif
#include "paddle/phi/backends/dynload/port.h"

namespace phi {

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

static constexpr const char* kDefaultThreadName = "unnamed";
// Returns kDefaultThreadName if SetCurrentThreadName is never called.
std::string GetCurrentThreadName();

// Return the map from StdTid to ThreadName
// Returns current snapshot of all threads. Make sure there is no thread
// create/destory when using it.
std::unordered_map<uint64_t, std::string> GetAllThreadNames();

// Thread name is immutable, only the first call will succeed.
// Returns false on failure.
bool SetCurrentThreadName(const std::string& name);

uint32_t GetProcessId();

}  // namespace phi
