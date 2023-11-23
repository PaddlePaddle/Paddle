/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/gcu_device.h"

#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {

int GetNumDevices() {
  int count;
  RT_CHECK(topsGetDeviceCount(&count));
  return count;
}

std::vector<int> GetDeviceIds() {
  int device_count = GetNumDevices();
  std::vector<int> devices(device_count, 0);
  for (int i = 0; i < device_count; ++i) {
    devices[i] = i;
  }
  return devices;
}

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
