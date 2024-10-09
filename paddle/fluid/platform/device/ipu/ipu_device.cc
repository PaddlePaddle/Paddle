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

#include <popart/devicemanager.hpp>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {

namespace {
const bool GetBoolEnv(const std::string& str) {
  char* str_val = getenv(str.c_str());
  if (str_val == NULL) {
    return false;
  } else {
    bool val = false;
    if (strcmp(str_val, "1") == 0 || strcmp(str_val, "true") == 0 ||
        strcmp(str_val, "True") == 0 || strcmp(str_val, "TRUE") == 0)
      val = true;
    return val;
  }
}
}  // namespace

int GetNumDevices() {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  bool compile_only = GetBoolEnv("IPU_COMPILE_ONLY");
  if (ipu_model || compile_only) {
    return 1;
  }
  int num_devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices().size();
  PADDLE_ENFORCE_GT(
      num_devices,
      0,
      common::errors::Unavailable("Do not found any IPU devices, please "
                                  "make sure Poplar sdk is enabled"));
  return num_devices;
}

std::vector<int> GetDeviceIds() {
  bool ipu_model = GetBoolEnv("POPLAR_IPUMODEL");
  bool compile_only = GetBoolEnv("IPU_COMPILE_ONLY");
  if (ipu_model || compile_only) {
    return {0};
  }
  std::vector<int> device_ids;
  auto devices =
      popart::DeviceManager::createDeviceManager().enumerateDevices();
  PADDLE_ENFORCE_GT(
      devices.size(),
      0,
      common::errors::Unavailable("Do not found any IPU devices, please make "
                                  "sure Poplar sdk is enabled."));
  for (auto device : devices) {
    device_ids.push_back(device->getId());
  }
  return device_ids;
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
