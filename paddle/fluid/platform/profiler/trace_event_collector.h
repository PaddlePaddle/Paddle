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
#include "paddle/phi/api/profiler/trace_event_collector.h"

namespace paddle {
namespace platform {

class TraceEventCollector : public phi::TraceEventCollector {
 public:
  void AddOperatorSupplementEvent(OperatorSupplementEvent&& event) {
    op_supplement_events_.push_back(event);
  }

  const std::list<OperatorSupplementEvent>& OperatorSupplementEvents() const {
    return op_supplement_events_;
  }

  void ClearAll() {
    thread_names_.clear();
    host_events_.clear();
    runtime_events_.clear();
    device_events_.clear();
    mem_events_.clear();
    op_supplement_events_.clear();
  }

 private:
  std::list<OperatorSupplementEvent> op_supplement_events_;
};

}  // namespace platform
}  // namespace paddle
