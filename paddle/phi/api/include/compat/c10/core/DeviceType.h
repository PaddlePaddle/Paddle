// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// If you modified DeviceType in this file, please also sync your changes into
// torch/headeronly/core/DeviceType.h.
#include <torch/headeronly/core/DeviceType.h>

#include <ostream>

#include "paddle/phi/common/place.h"

namespace c10 {

inline phi::AllocationType DeviceTypeToPhi(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
      return phi::AllocationType::CPU;
    case DeviceType::CUDA:
      return phi::AllocationType::GPU;
    case DeviceType::XPU:
      return phi::AllocationType::XPU;
    case DeviceType::IPU:
      return phi::AllocationType::IPU;
    case DeviceType::CUSTOM:
      return phi::AllocationType::CUSTOM;
  }
  return phi::AllocationType::UNDEFINED;
}

inline DeviceType PhiToDeviceType(phi::AllocationType d) {
  switch (d) {
    case phi::AllocationType::CPU:
      return DeviceType::CPU;
    case phi::AllocationType::GPU:
      return DeviceType::CUDA;
    case phi::AllocationType::XPU:
      return DeviceType::XPU;
    case phi::AllocationType::IPU:
      return DeviceType::IPU;
    case phi::AllocationType::CUSTOM:
      return DeviceType::CUSTOM;
    default:
      return DeviceType::CPU;
  }
}

inline bool isValidDeviceType(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
    case DeviceType::CUDA:
    case DeviceType::XPU:
    case DeviceType::IPU:
    case DeviceType::CUSTOM:
      return true;
    default:
      return false;
  }
}

inline std::ostream& operator<<(std::ostream& os, DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
      os << "cpu";
      break;
    case DeviceType::CUDA:
      os << "cuda";
      break;
    case DeviceType::XPU:
      os << "xpu";
      break;
    case DeviceType::IPU:
      os << "ipu";
      break;
    case DeviceType::CUSTOM:
      os << "privateuseone";
      break;
  }
  return os;
}

}  // namespace c10
