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
                          static_cast<size_t>(5)>
      types = {{
          {"cpu", DeviceType::CPU},
          {"cuda", DeviceType::CUDA},
          {"gpu", DeviceType::GPU},
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
      "'cpu', 'cuda', 'gpu', 'ipu', and 'xpu'.",
      device_string));
}

Device::Device(const std::string& device_string)
    : inner_(phi::Place(phi::AllocationType::CPU, -1)) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");
  auto colon_pos = device_string.find(':');
  std::string type_str = colon_pos == std::string::npos
                             ? device_string
                             : device_string.substr(0, colon_pos);

  // Convert "gpu" to "cuda" for PyTorch compatibility
  if (type_str == "gpu") {
    type_str = "cuda";
  }

  DeviceType type = parse_type(type_str);

  // 默认 index = -1 (PyTorch 兼容)
  DeviceIndex index = -1;

  if (colon_pos != std::string::npos) {
    std::string index_str = device_string.substr(colon_pos + 1);
    try {
      index = static_cast<DeviceIndex>(std::stoi(index_str));
    } catch (const std::invalid_argument&) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Invalid device index: '%s' is not a number.", index_str));
    } catch (const std::out_of_range&) {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "Invalid device index: '%s' is out of range.", index_str));
    }
  }

  // 只有显式指定了 index 才设置
  if (index >= 0) {
    inner_ = phi::Place(c10DeviceTypeToPhiAllocationType(type), index);
  } else {
    // 对于没有显式 index 的情况，根据 PyTorch 行为：
    // - CPU: index = -1, has_index() = false
    // - CUDA: index = 0, has_index() = true
    if (type == DeviceType::CPU) {
      inner_ = phi::Place(c10DeviceTypeToPhiAllocationType(type), -1);
    } else {
      // 非 CPU 设备默认 index=0
      inner_ = phi::Place(c10DeviceTypeToPhiAllocationType(type), 0);
    }
  }
}

std::string Device::str() const {
  // Convert to PyTorch-compatible string format
  // CPU -> "cpu", GPU/CUDA -> "cuda"
  std::string str;
  if (type() == DeviceType::CPU) {
    str = "cpu";
  } else if (type() == DeviceType::CUDA || type() == DeviceType::GPU) {
    str = "cuda";
  } else if (type() == DeviceType::XPU) {
    str = "xpu";
  } else if (type() == DeviceType::IPU) {
    str = "ipu";
  } else {
    str = "unknown";
  }
  // Only add index if it's non-zero (PyTorch doesn't show :0 for default)
  if (has_index() && index() != 0) {
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
