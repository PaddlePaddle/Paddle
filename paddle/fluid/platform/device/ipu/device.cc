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

#include "paddle/fluid/platform/device/ipu/device.h"

namespace paddle {
namespace platform {
namespace ipu {

Device::Device(const popart::DeviceInfo& device_info)
    : id_(device_info.getId()), is_attached_(device_info.isAttached()) {
  popart::DeviceType popart_device_type = device_info.getType();
  switch (popart_device_type) {
    case popart::DeviceType::IpuModel:
      device_type_ = DeviceType::IpuModel;
      break;
    case popart::DeviceType::Ipu:
      device_type_ = DeviceType::Ipu;
      break;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "popart::DeviceType:Unsupported type %d", popart_device_type));
  }
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
