// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/device/device_manager.h"

namespace paddle {
namespace platform {

class DeviceGuard {
 public:
  explicit inline DeviceGuard(const Place& place)
      : dev_type_(PlaceHelper::GetDeviceType(place)) {
    prev_id = DeviceManager::GetDevice(dev_type_);
    cur_id = PlaceHelper::GetDeviceId(place);

    if (cur_id != prev_id) {
      DeviceManager::SetDevice(dev_type_, cur_id);
    }
  }

  inline ~DeviceGuard() {
    if (cur_id != prev_id) {
      DeviceManager::SetDevice(dev_type_, prev_id);
    }
  }

  DeviceGuard(const DeviceGuard& o) = delete;
  DeviceGuard& operator=(const DeviceGuard& o) = delete;

 private:
  size_t prev_id, cur_id;
  std::string dev_type_;
};

}  // namespace platform
}  // namespace paddle
