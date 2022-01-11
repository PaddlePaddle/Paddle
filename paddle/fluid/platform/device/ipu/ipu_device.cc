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

#include "paddle/fluid/platform/device/ipu/ipu_device.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

int GetNumDevices() {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    return 1;
  }
  int num_devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices().size();
  PADDLE_ENFORCE_GT(num_devices, 0, platform::errors::Unavailable(
                                        "Do not found any IPU devices, please "
                                        "make sure Poplar sdk is enabled"));
  return num_devices;
}

std::vector<int> GetDeviceIds() {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  if (ipu_model) {
    return {0};
  }
  std::vector<int> device_ids;
  auto devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices();
  PADDLE_ENFORCE_GT(
      devices.size(), 0,
      platform::errors::Unavailable("Do not found any IPU devices, please make "
                                    "sure Poplar sdk is enabled."));
  for (auto device : devices) {
    device_ids.push_back(device->getId());
  }
  return device_ids;
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
