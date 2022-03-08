// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int dev_id) { SetDeviceIndex(dev_id); }

  explicit CUDADeviceGuard(const CUDAPlace& place)
      : CUDADeviceGuard(place.device) {}

  // create uninitialized CUDADeviceGuard
  CUDADeviceGuard() {}

  ~CUDADeviceGuard() {
    if (prev_id_ != -1) {
      platform::SetDeviceId(prev_id_);
    }
  }

  inline void SetDeviceIndex(const int dev_id) {
    int prev_id = platform::GetCurrentDeviceId();
    if (prev_id != dev_id) {
      prev_id_ = prev_id;
      platform::SetDeviceId(dev_id);
    }
  }

  void SetDevice(const CUDAPlace& place) {
    int dev_id = place.device;
    SetDeviceIndex(dev_id);
  }

  CUDADeviceGuard(const CUDADeviceGuard& o) = delete;
  CUDADeviceGuard& operator=(const CUDADeviceGuard& o) = delete;

 private:
  int prev_id_{-1};
};

}  // namespace platform
}  // namespace paddle
