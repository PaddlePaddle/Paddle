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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <array>
#include "paddle/common/enforce.h"

namespace c10 {

DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<const char*, DeviceType>,
                          static_cast<size_t>(4)>
      types = {{
          {"cpu", DeviceType::CPU},
          {"cuda", DeviceType::CUDA},
          {"ipu", DeviceType::IPU},
          {"xpu", DeviceType::XPU},
      }};
  for (const auto& type_pair : types) {
    if (device_string == type_pair.first) {
      return type_pair.second;
    }
  }
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Unknown device type: '%s'. Supported device types are ",
      "'cpu', 'cuda', 'ipu', and 'xpu'.",
      device_string));
}

Device::Device(const std::string& device_string) : Device(Type::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");
  auto colon_pos = device_string.find(':');
  std::string type_str = colon_pos == std::string::npos
                             ? device_string
                             : device_string.substr(0, colon_pos);
  DeviceType type = parse_type(type_str);
  DeviceIndex index = 0;
  if (colon_pos != std::string::npos) {
    std::string index_str = device_string.substr(colon_pos + 1);
    try {
      index = static_cast<DeviceIndex>(std::stoi(index_str));
      inner_ = phi::Place(type, index);
    } catch (const std::invalid_argument&) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Invalid device index: '%s' is not a number.", index_str));
    } catch (const std::out_of_range&) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Invalid device index: '%s' is out of range.", index_str));
    }
  }
}

std::string Device::str() const {
  std::string str = phi::AllocationTypeStr(type());
  if (has_index()) {
    str.push_back(':');
    str.append(std::to_string(index()));
  }
  return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device) {
  stream << device.str();
  return stream;
}
}  // namespace c10
