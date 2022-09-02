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

#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/event.h"  // import EventRole, TODO(TIEXING): remove later
#include "paddle/fluid/platform/profiler/trace_event.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace platform {

struct CommonEvent {
 public:
  CommonEvent(const char *name,
              uint64_t start_ns,
              uint64_t end_ns,
              EventRole role,
              TracerEventType type)
      : name(name),
        start_ns(start_ns),
        end_ns(end_ns),
        role(role),
        type(type) {}

  CommonEvent(std::function<void *(size_t)> arena_allocator,
              const std::string &name_str,
              uint64_t start_ns,
              uint64_t end_ns,
              EventRole role,
              TracerEventType type,
              const std::string &attr_str)
      : start_ns(start_ns), end_ns(end_ns), role(role), type(type) {
    auto buf = static_cast<char *>(arena_allocator(name_str.length() + 1));
    strncpy(buf, name_str.c_str(), name_str.length() + 1);
    name = buf;
    buf = static_cast<char *>(arena_allocator(attr_str.length() + 1));
    strncpy(buf, attr_str.c_str(), attr_str.length() + 1);
    attr = buf;
  }

  CommonEvent(std::function<void *(size_t)> arena_allocator,
              const std::string &name_str,
              uint64_t start_ns,
              uint64_t end_ns,
              EventRole role,
              TracerEventType type)
      : start_ns(start_ns), end_ns(end_ns), role(role), type(type) {
    auto buf = static_cast<char *>(arena_allocator(name_str.length() + 1));
    strncpy(buf, name_str.c_str(), name_str.length() + 1);
    name = buf;
  }

  const char *name = nullptr;  // not owned, designed for performance
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  EventRole role = EventRole::kOrdinary;
  TracerEventType type = TracerEventType::NumTypes;
  const char *attr = nullptr;  // not owned, designed for performance
};

struct CommonMemEvent {
 public:
  CommonMemEvent(uint64_t timestamp_ns,
                 uint64_t addr,
                 TracerMemEventType type,
                 int64_t increase_bytes,
                 const Place &place,
                 uint64_t current_allocated,
                 uint64_t current_reserved,
                 uint64_t peak_allocated,
                 uint64_t peak_reserved)
      : timestamp_ns(timestamp_ns),
        addr(addr),
        type(type),
        increase_bytes(increase_bytes),
        place(place),
        current_allocated(current_allocated),
        current_reserved(current_reserved),
        peak_allocated(peak_allocated),
        peak_reserved(peak_reserved) {}
  uint64_t timestamp_ns;
  uint64_t addr;
  TracerMemEventType type;
  int64_t increase_bytes;
  Place place;
  uint64_t current_allocated;
  uint64_t current_reserved;
  uint64_t peak_allocated;
  uint64_t peak_reserved;
};

struct OperatorSupplementOriginEvent {
 public:
  OperatorSupplementOriginEvent(
      std::function<void *(size_t)> arena_allocator,
      uint64_t timestamp_ns,
      const std::string &type_name,
      const std::map<std::string, std::vector<framework::DDim>> &input_shapes,
      const std::map<std::string, std::vector<framework::proto::VarType::Type>>
          &dtypes,
      const std::vector<std::string> callstack)
      : timestamp_ns(timestamp_ns),
        input_shapes(input_shapes),
        dtypes(dtypes),
        callstack(callstack) {
    auto buf = static_cast<char *>(arena_allocator(type_name.length() + 1));
    strncpy(buf, type_name.c_str(), type_name.length() + 1);
    op_type = buf;
  }
  OperatorSupplementOriginEvent(
      std::function<void *(size_t)> arena_allocator,
      uint64_t timestamp_ns,
      const std::string &type_name,
      const std::vector<std::pair<const char *, std::vector<framework::DDim>>>
          &shapes,
      const std::map<std::string, std::vector<framework::proto::VarType::Type>>
          &dtypes,
      const std::vector<std::string> callstack)
      : timestamp_ns(timestamp_ns), dtypes(dtypes), callstack(callstack) {
    auto buf = static_cast<char *>(arena_allocator(type_name.length() + 1));
    strncpy(buf, type_name.c_str(), type_name.length() + 1);
    op_type = buf;
    for (auto it = shapes.begin(); it != shapes.end(); it++) {
      input_shapes[std::string((*it).first)] = (*it).second;
    }
  }
  uint64_t timestamp_ns;
  const char *op_type = nullptr;  // not owned, designed for performance
  // input shapes
  std::map<std::string, std::vector<framework::DDim>> input_shapes;
  std::map<std::string, std::vector<framework::proto::VarType::Type>> dtypes;
  // call stack
  const std::vector<std::string> callstack;
};

}  // namespace platform
}  // namespace paddle
