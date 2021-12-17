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

#include "paddle/pten/common/device.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/pten/api/ext/exception.h"

namespace paddle {
namespace experimental {

const char* DeviceTypeStr(DeviceType type) {
  switch (type) {
    case DeviceType::kUndef:
      return "kUndef";
    case DeviceType::kHost:
      return "kUndef";
    case DeviceType::kXpu:
      return "kXpu";
    case DeviceType::kCuda:
      return "kCuda";
    case DeviceType::kHip:
      return "kHip";
    case DeviceType::kNpu:
      return "kNpu";
    default:
      PD_THROW("Invalid pten device type.");
  }
  return {};
}

Device::Device(DeviceType type, int8_t id) : type_(type), id_(id) {
  PADDLE_ENFORCE_GE(
      id,
      0,
      platform::errors::InvalidArgument(
          "The device id needs to start from zero, but you passed in %d.", id));
}

Device::Device(DeviceType type) : type_(type), id_(0) {
  PADDLE_ENFORCE_EQ(
      type,
      DeviceType::kHost,
      platform::errors::InvalidArgument(
          "The device id needs to start from zero, but you passed in %s.",
          DeviceTypeStr(type)));
}

std::string Device::DebugString() const {
  std::string str{"DeviceType:"};
  return str + DeviceTypeStr(type_) + ", id: " + std::to_string(id_);
}

}  // namespace experimental
}  // namespace paddle
