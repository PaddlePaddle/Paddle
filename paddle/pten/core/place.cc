/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/place.h"

#include <string>

#include "paddle/pten/api/ext/exception.h"

namespace pten {

const char *AllocationTypeStr(AllocationType type) {
  switch (type) {
    case AllocationType::kUndef:
      return "kUndef";
    case AllocationType::kCpu:
      return "kCpu";
    case AllocationType::kGpu:
      return "kGpu";
    case AllocationType::kGpuPinned:
      return "kGpuPinned";
    case AllocationType::kXpu:
      return "kXpu";
    case AllocationType::kNpu:
      return "kNpu";
    case AllocationType::kNpuPinned:
      return "kNpuPinned";
    case AllocationType::kIpu:
      return "kIpu";
    case AllocationType::kMlu:
      return "kMlu";
    default:
      PD_THROW("Invalid pten device type.");
      return {};
  }
}

std::string Place::DebugString() const {
  std::string str{"Place:{type = "};
  return str + AllocationTypeStr(alloc_type_) + ", id = " +
         std::to_string(device) + "}";
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  os << p.DebugString();
  return os;
}

}  // namespace pten
