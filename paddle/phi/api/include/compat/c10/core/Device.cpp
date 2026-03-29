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

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <string>

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
          {"ipu", DeviceType::IPU},
          {"xpu", DeviceType::XPU},
          {"privateuseone", DeviceType::PrivateUse1},
      }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [&device_string](const std::pair<const char*, DeviceType>& p) {
        return p.first && p.first == device_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  TORCH_CHECK(false,
              "Expected one of cpu, cuda, ipu, xpu, privateuseone device type "
              "at start of device string: ",
              device_string);
}

enum DeviceStringParsingState { kStart, kIndexStart, kIndexRest, kError };

Device::Device(const std::string& device_string) : Device(Type::CPU) {
  TORCH_CHECK(!device_string.empty(), "Device string must not be empty");

  std::string device_name, device_index_str;
  DeviceStringParsingState pstate = DeviceStringParsingState::kStart;

  for (size_t i = 0;
       pstate != DeviceStringParsingState::kError && i < device_string.size();
       ++i) {
    const char ch = device_string.at(i);
    const unsigned char uch = static_cast<unsigned char>(ch);
    switch (pstate) {
      case DeviceStringParsingState::kStart:
        if (ch != ':') {
          if (std::isalpha(uch) || ch == '_') {
            device_name.push_back(ch);
          } else {
            pstate = DeviceStringParsingState::kError;
          }
        } else {
          pstate = DeviceStringParsingState::kIndexStart;
        }
        break;
      case DeviceStringParsingState::kIndexStart:
        if (std::isdigit(uch)) {
          device_index_str.push_back(ch);
          pstate = DeviceStringParsingState::kIndexRest;
        } else {
          pstate = DeviceStringParsingState::kError;
        }
        break;
      case DeviceStringParsingState::kIndexRest:
        if (device_index_str.at(0) == '0') {
          pstate = DeviceStringParsingState::kError;
          break;
        }
        if (std::isdigit(uch)) {
          device_index_str.push_back(ch);
        } else {
          pstate = DeviceStringParsingState::kError;
        }
        break;
      case DeviceStringParsingState::kError:
        break;
    }
  }

  const bool has_error = device_name.empty() ||
                         pstate == DeviceStringParsingState::kError ||
                         (pstate == DeviceStringParsingState::kIndexStart &&
                          device_index_str.empty());
  TORCH_CHECK(!has_error, "Invalid device string: '", device_string, "'");

  try {
    if (!device_index_str.empty()) {
      index_ = static_cast<DeviceIndex>(std::stoi(device_index_str));
    }
  } catch (const std::exception&) {
    TORCH_CHECK(false,
                "Could not parse device index '",
                device_index_str,
                "' in device string '",
                device_string,
                "'");
  }
  type_ = parse_type(device_name);
  validate();
}

std::string Device::str() const {
  std::string str = DeviceTypeToString(type());
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
