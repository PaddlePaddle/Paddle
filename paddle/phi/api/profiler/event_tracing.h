/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/phi/api/profiler/event.h"
#include "paddle/phi/api/profiler/trace_event.h"
#include "paddle/utils/test_macros.h"

namespace phi {

// Default tracing level.
// It is Recommended to set the level explicitly.
static constexpr uint32_t kDefaultTraceLevel = 4;

// Host event tracing. A trace starts when an object of this class is created
// and stops when the object is destroyed.
// Chrome Trace Viewer Format: Duration Event/Complete Event
class TEST_API RecordEvent {
 public:
  static bool IsEnabled();
  /**
   * @param name If your string argument has a longer lifetime (e.g.: string
   * literal, static variables, etc) than the event, use 'const char* name'.
   * Do your best to avoid using 'std::string' as the argument type. It will
   * cause deep-copy to harm performance.
   * @param type Classification which is used to instruct the profiling
   * data statistics.
   * @param level Used to filter events, works like glog VLOG(level).
   * RecordEvent will works if HostTraceLevel >= level.
   */
  explicit RecordEvent(
      const std::string& name,
      const TracerEventType type = TracerEventType::UserDefined,
      uint32_t level = kDefaultTraceLevel,
      const EventRole role = EventRole::kOrdinary);

  /**
   * @param name It is the caller's responsibility to manage the underlying
   * storage. RecordEvent stores the pointer.
   * @param type Classification which is used to instruct the profiling
   * data statistics.
   * @param level Used to filter events, works like glog VLOG(level).
   * RecordEvent will works if HostTraceLevel >= level.
   */
  explicit RecordEvent(
      const char* name,
      const TracerEventType type = TracerEventType::UserDefined,
      uint32_t level = kDefaultTraceLevel,
      const EventRole role = EventRole::kOrdinary);

  RecordEvent(const std::string& name,
              const std::string& attr,
              const TracerEventType type = TracerEventType::UserDefined,
              uint32_t level = kDefaultTraceLevel,
              const EventRole role = EventRole::kOrdinary);

  // Stop event tracing explicitly before the object goes out of scope.
  // Sometimes it's inconvenient to use RAII
  void End();

  ~RecordEvent() { End(); }

 private:
  void OriginalConstruct(const std::string& name,
                         const EventRole role,
                         const std::string& attr);

  bool is_enabled_{false};
  bool is_pushed_{false};
  // Event name
  std::string* name_{nullptr};
  const char* shallow_copy_name_{nullptr};
  uint64_t start_ns_;
  // Need to distinguish name by op type, block_id, program_id and perhaps
  // different kernel invocations within an op.
  // std::string full_name_;
  EventRole role_{EventRole::kOrdinary};
  TracerEventType type_{TracerEventType::UserDefined};
  std::string* attr_{nullptr};
  bool finished_{false};
};

}  // namespace phi
