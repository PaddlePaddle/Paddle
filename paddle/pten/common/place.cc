/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/common/place.h"

#include <sstream>
#include <string>

#include "paddle/pten/api/ext/exception.h"

namespace pten {

const char *AllocationTypeStr(AllocationType type) {
  switch (type) {
    case AllocationType::UNDEFINED:
      return "undefined";
    case AllocationType::CPU:
      return "cpu";
    case AllocationType::GPU:
      return "gpu";
    case AllocationType::GPUPINNED:
      return "gpu_pinned";
    case AllocationType::XPU:
      return "xpu";
    case AllocationType::NPU:
      return "npu";
    case AllocationType::NPUPINNED:
      return "npu_pinned";
    case AllocationType::IPU:
      return "ipu";
    case AllocationType::MLU:
      return "mlu";
    default:
      PD_THROW("Invalid pten device type.");
      return {};
  }
}

std::string Place::DebugString() const {
  std::ostringstream os;
  os << "Place(";
  os << AllocationTypeStr(alloc_type_);
  if (alloc_type_ == AllocationType::GPUPINNED ||
      alloc_type_ == AllocationType::NPUPINNED ||
      alloc_type_ == AllocationType::CPU) {
    os << ")";
  } else {
    os << ":" << std::to_string(device) << ")";
  }
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  os << p.DebugString();
  return os;
}

}  // namespace pten
