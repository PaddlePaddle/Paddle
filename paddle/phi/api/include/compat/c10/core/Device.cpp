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

namespace {

const char* DeviceTypeToString(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    case DeviceType::XPU:
      return "xpu";
    case DeviceType::IPU:
      return "ipu";
    case DeviceType::CUSTOM:
      return "privateuseone";
  }
  return "cpu";
}

}  // namespace

DeviceType parse_type(const std::string& device_string) {
  static const std::array<std::pair<const char*, DeviceType>,
                          static_cast<size_t>(5)>
      types = {{
          {"cpu", DeviceType::CPU},
          {"cuda", DeviceType::CUDA},
          {"xpu", DeviceType::XPU},
          {"ipu", DeviceType::IPU},
          {"custom", DeviceType::CUSTOM},
      }};
  for (const auto& type_pair : types) {
    if (device_string == type_pair.first) {
      return type_pair.second;
    }
  }
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Unknown device type: '%s'. Supported device types are "
      "'cpu', 'cuda', 'xpu', 'ipu', and 'custom'.",
      device_string));
}

Device::Device(const std::string& device_string)
    : inner_(phi::Place(phi::AllocationType::CPU, 0)), has_index_(false) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");
  auto colon_pos = device_string.find(':');
  std::string type_str = colon_pos == std::string::npos
                             ? device_string
                             : device_string.substr(0, colon_pos);

  DeviceType type = parse_type(type_str);

  // 解析 index
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

  // 固定内存类型（GPUPINNED/XPUPINNED）不携带有效的 device index。
  // 其余设备遵循 PyTorch 语义：index==-1 表示无显式 index。
  const phi::AllocationType alloc_type = c10DeviceTypeToPhiAllocationType(type);
  const bool no_index_type =
      (type == DeviceType::GPUPINNED || type == DeviceType::XPUPINNED);
  has_index_ = !no_index_type && (index != -1);
  inner_ = phi::Place(alloc_type, has_index_ ? index : 0);
}

std::string Device::str() const {
  std::string str;
  switch (type()) {
    case DeviceType::CPU:
      str = "cpu";
      break;
    case DeviceType::CUDA:
      str = "cuda";
      break;
    case DeviceType::GPUPINNED:
      // GPU 固定内存在物理上位于 CPU 侧，字符串表示与 PyTorch 保持一致
      str = "cpu";
      break;
    case DeviceType::XPU:
      str = "xpu";
      break;
    case DeviceType::XPUPINNED:
      str = "xpu";
      break;
    case DeviceType::IPU:
      str = "ipu";
      break;
    case DeviceType::CUSTOM:
      str = "custom";
      break;
    default:
      str = "unknown";
      break;
  }
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
