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

#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler/trace_event.h"

namespace paddle {
namespace platform {
// Memory event tracing. A trace marks memory manipulation such as allocation
// and free.
// The events can be used to draw memory variation curve.
class RecordMemEvent {
 public:
  /**
   * @param ptr:  Pointer address allocated or free.
   * @param place: Device for this memory event.
   * @param size: Memory size allocated or free.
   * @param type: Denote manipulation type for this memory event.
   */
  explicit RecordMemEvent(
      const void* ptr,
      const Place& place,
      size_t size,
      const TracerMemEventType type = TracerMemEventType::Allocate);
};

}  // namespace platform
}  // namespace paddle
