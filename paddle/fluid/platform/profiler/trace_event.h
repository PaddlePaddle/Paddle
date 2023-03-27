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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/type_defs.h"
#include "paddle/phi/api/profiler/trace_event.h"

namespace paddle {
namespace platform {

using TracerEventType = phi::TracerEventType;
using TracerMemEventType = phi::TracerMemEventType;
using KernelEventInfo = phi::KernelEventInfo;
using MemcpyEventInfo = phi::MemcpyEventInfo;
using MemsetEventInfo = phi::MemsetEventInfo;
using HostTraceEvent = phi::HostTraceEvent;
using RuntimeTraceEvent = phi::RuntimeTraceEvent;
using DeviceTraceEvent = phi::DeviceTraceEvent;
using MemTraceEvent = phi::MemTraceEvent;

struct OperatorSupplementEvent {
  OperatorSupplementEvent() = default;
  OperatorSupplementEvent(
      uint64_t timestamp_ns,
      const std::string& op_type,
      const std::map<std::string, std::vector<std::vector<int64_t>>>&
          input_shapes,
      const std::map<std::string, std::vector<std::string>>& dtypes,
      const std::string& callstack,
      const framework::AttributeMap& attributes,
      uint64_t op_id,
      uint64_t process_id,
      uint64_t thread_id)
      : timestamp_ns(timestamp_ns),
        op_type(op_type),
        input_shapes(input_shapes),
        dtypes(dtypes),
        callstack(callstack),
        attributes(attributes),
        op_id(op_id),
        process_id(process_id),
        thread_id(thread_id) {}
  // timestamp of the record
  uint64_t timestamp_ns;
  // op type name
  std::string op_type;
  // input shapes
  std::map<std::string, std::vector<std::vector<int64_t>>> input_shapes;
  std::map<std::string, std::vector<std::string>> dtypes;
  // call stack
  std::string callstack;
  // op attributes
  framework::AttributeMap attributes;
  // op id
  uint64_t op_id;
  // process id of the record
  uint64_t process_id;
  // thread id of the record
  uint64_t thread_id;
};

}  // namespace platform
}  // namespace paddle
