// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>
#include <functional>
#include <string>

#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler/trace_event.h"

namespace paddle {
namespace platform {

struct CommonMemEvent {
 public:
  CommonMemEvent(std::function<void *(size_t)> arena_allocator, uint64_t id,
                 uint64_t timestamp_ns, uint64_t addr, TracerMemEventType type,
                 int64_t increase_bytes, const std::string &place,
                 uint64_t current_allocated, uint64_t current_reserved)
      : id(id),
        timestamp_ns(timestamp_ns),
        addr(addr),
        type(type),
        increase_bytes(increase_bytes) current_allocated(current_allocated)
            current_reserved(current_reserved) {
    auto buf = static_cast<char *>(arena_allocator(place.length() + 1));
    strncpy(buf, place.c_str(), place.length() + 1);
    place = buf;
  }

  uint64_t id;  // not owned, designed for performance
  uint64_t timestamp_ns;
  uint64_t addr;
  TracerMemEventType type;
  int64_t increase_bytes;
  const char *place = nullptr;
  uint64_t current_allocated;
  uint64_t current_reserved;
};

}  // namespace platform
}  // namespace paddle
