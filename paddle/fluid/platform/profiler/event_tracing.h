/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/event.h"

namespace paddle {
namespace platform {

// CPU event tracing. A trace marks something that happens but has no duration
// associated with it. For example, thread starts working.
// Chrome Trace Viewer Format: Instant Event
struct RecordInstantEvent {
  explicit RecordInstantEvent(const char* name,
                              const EventRole role = EventRole::kOrdinary);
};

// CPU event tracing. A trace starts when an object of this clas is created and
// stops when the object is destroyed.
// Chrome Trace Viewer Format: Duration Event/Complte Event
class RecordEvent {
 public:
  explicit RecordEvent(const std::string& name,
                       const EventRole role = EventRole::kOrdinary);

  explicit RecordEvent(const char* name,
                       const EventRole role = EventRole::kOrdinary);

  RecordEvent(const std::string& name, const EventRole role,
              const std::string& attr);

  // Stop event tracing explicitly before the object goes out of scope.
  // Sometimes it's inconvenient to use RAII
  void End();

  ~RecordEvent() { End(); }

 private:
  void OriginalConstruct(const std::string& name, const EventRole role,
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
  std::string* attr_{nullptr};
  bool finished_{false};
};

}  // namespace platform
}  // namespace paddle
