// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/trace_event.h"

namespace paddle {
namespace platform {

// Host event tracing. A trace marks something that happens but has no duration
// associated with it. For example, thread starts working.
// Chrome Trace Viewer Format: Instant Event
struct RecordInstantEvent {
  /**
   * @param name: It is the caller's responsibility to manage the underlying
   * storage. RecordInstantEvent stores the pointer.
   * @param type: Classification which is used to instruct the profiling
   * data statistics.
   * @param level: Used to filter events, works like glog VLOG(level).
   * RecordEvent will works if HostTraceLevel >= level.
   */
  explicit RecordInstantEvent(const char* name,
                              phi::TracerEventType type,
                              uint32_t level = phi::kDefaultTraceLevel);
};

using RecordEvent = phi::RecordEvent;

}  // namespace platform
}  // namespace paddle
