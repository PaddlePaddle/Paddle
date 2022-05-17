// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <type_traits>
#include <vector>
#include "paddle/fluid/framework/new_executor/workqueue/thread_data_registry.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/os_info.h"
#include "paddle/fluid/platform/profiler/common_mem_event.h"
#include "paddle/fluid/platform/profiler/host_event_recorder.h"

namespace paddle {
namespace platform {

struct ThreadMemEventSection {
  std::string thread_name;
  uint64_t thread_id;
  std::vector<CommonMemEvent> events;
};

class ThreadMemEventRecorder {
 public:
  ThreadMemEventRecorder() {
    thread_id_ = GetCurrentThreadSysId();
    thread_name_ = GetCurrentThreadName();
  }

  DISABLE_COPY_AND_ASSIGN(ThreadMemEventRecorder);

 public:
  // Forward call to EventContainer::Record
  template <typename... Args>
  void RecordMemEvent(Args &&... args) {
    base_evt_cntr_.Record(std::forward<Args>(args)...);
  }

  ThreadMemEventSection GatherEvents() {
    ThreadMemEventSection thr_sec;
    thr_sec.thread_name = thread_name_;
    thr_sec.thread_id = thread_id_;
    thr_sec.events = std::move(base_evt_cntr_.Reduce());
    return thr_sec;
  }

 private:
  uint64_t thread_id_;
  std::string thread_name_;
  EventContainer<CommonMemEvent> base_evt_cntr_;
};

struct HostMemEventSection {
  std::string process_name;
  uint64_t process_id;
  std::vector<ThreadMemEventSection> thr_sections;
};

class HostMemEventRecorder {
 public:
  // singleton
  static HostMemEventRecorder &GetInstance() {
    static HostMemEventRecorder instance;
    return instance;
  }

  // thread-safe
  // If your string argument has a longer lifetime than the Event,
  // use 'const char*'. e.g.: string literal, op name, etc.
  // Do your best to avoid using 'std::string' as the argument type.
  // It will cause deep-copy to harm performance.
  template <typename... Args>
  void RecordMemEvent(Args &&... args) {
    GetThreadLocalRecorder()->RecordMemEvent(mem_event_indx,
                                             std::forward<Args>(args)...);
    HostEventInfoSupplement::GetInstance().RecordMemoryInfo(mem_event_indx);
    mem_event_indx += 1;
  }

  // thread-unsafe, make sure make sure there is no running tracing.
  // Poor performance, call it at the ending
  HostMemEventSection GatherEvents() {
    auto thr_recorders =
        ThreadMemEventRecorderRegistry::GetInstance().GetAllThreadDataByRef();
    HostMemEventSection host_sec;
    host_sec.process_id = GetProcessId();
    host_sec.thr_sections.reserve(thr_recorders.size());
    for (auto &kv : thr_recorders) {
      auto &thr_recorder = kv.second.get();
      host_sec.thr_sections.emplace_back(
          std::move(thr_recorder.GatherEvents()));
    }
    return host_sec;
  }

 private:
  using ThreadMemEventRecorderRegistry =
      framework::ThreadDataRegistry<ThreadMemEventRecorder>;

  HostMemEventRecorder() = default;
  DISABLE_COPY_AND_ASSIGN(HostMemEventRecorder);

  ThreadMemEventRecorder *GetThreadLocalRecorder() {
    return ThreadMemEventRecorderRegistry::GetInstance()
        .GetMutableCurrentThreadData();
  }
  uint64_t mem_event_indx = 0;
};

}  // namespace platform
}  // namespace paddle
