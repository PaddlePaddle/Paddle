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
#include "paddle/phi/api/profiler/common_event.h"

namespace paddle {
namespace platform {

using CommonEvent = phi::CommonEvent;

using CommonMemEvent = phi::CommonMemEvent;

struct OperatorSupplementOriginEvent {
 public:
  OperatorSupplementOriginEvent(
      std::function<void *(size_t)> arena_allocator,
      uint64_t timestamp_ns,
      const std::string &type_name,
      const std::map<std::string, std::vector<framework::DDim>> &input_shapes,
      const std::map<std::string, std::vector<framework::proto::VarType::Type>>
          &dtypes,
      const framework::AttributeMap &attributes,
      uint64_t op_id)
      : timestamp_ns(timestamp_ns),
        input_shapes(input_shapes),
        dtypes(dtypes),
        attributes(attributes),
        op_id(op_id) {
    auto buf = static_cast<char *>(arena_allocator(type_name.length() + 1));
    strncpy(buf, type_name.c_str(), type_name.length() + 1);
    op_type = buf;
  }

  uint64_t timestamp_ns;
  const char *op_type = nullptr;  // not owned, designed for performance
  // input shapes
  std::map<std::string, std::vector<framework::DDim>> input_shapes;
  std::map<std::string, std::vector<framework::proto::VarType::Type>> dtypes;
  // op attributes
  framework::AttributeMap attributes;
  // op id
  uint64_t op_id;
};

}  // namespace platform
}  // namespace paddle
