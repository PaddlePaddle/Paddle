/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <list>
#include <string>
#include <unordered_map>
#include "paddle/fluid/platform/profiler/trace_event.h"

namespace paddle {
namespace platform {

class TraceEventCollector {
 public:
  void AddHostEvent(HostTraceEvent&& event) { host_events_.push_back(event); }

  void AddRuntimeEvent(RuntimeTraceEvent&& event) {
    runtime_events_.push_back(event);
  }

  void AddDeviceEvent(DeviceTraceEvent&& event) {
    device_events_.push_back(event);
  }

  void AddThreadName(uint64_t tid, const std::string& name) {
    thread_names_[tid] = name;
  }

  const std::list<HostTraceEvent>& HostEvents() const { return host_events_; }

  const std::list<RuntimeTraceEvent>& RuntimeEvents() const {
    return runtime_events_;
  }

  const std::list<DeviceTraceEvent>& DeviceEvents() const {
    return device_events_;
  }

  const std::unordered_map<uint64_t, std::string>& ThreadNames() const {
    return thread_names_;
  }

  void ClearAll() {
    thread_names_.clear();
    host_events_.clear();
    runtime_events_.clear();
    device_events_.clear();
  }

 private:
  std::unordered_map<uint64_t, std::string> thread_names_;
  std::list<HostTraceEvent> host_events_;
  std::list<RuntimeTraceEvent> runtime_events_;
  std::list<DeviceTraceEvent> device_events_;
};

}  // namespace platform
}  // namespace paddle
