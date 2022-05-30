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

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/npu_resource_pool.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"

namespace paddle {
namespace platform {

NpuStreamResourcePool::NpuStreamResourcePool() {
  int dev_cnt = platform::GetNPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetNPUDeviceId(dev_idx);
      aclrtStream stream;
      NPUStreamCreate(&stream);
      return stream;
    };

    auto deleter = [dev_idx](aclrtStream stream) {
      platform::SetNPUDeviceId(dev_idx);
      NPUStreamDestroy(stream);
    };

    pool_.emplace_back(ResourcePool<NpuStreamObject>::Create(creator, deleter));
  }
}

NpuStreamResourcePool& NpuStreamResourcePool::Instance() {
  static NpuStreamResourcePool pool;
  return pool;
}

std::shared_ptr<NpuStreamObject> NpuStreamResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx, 0,
      platform::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx, pool_.size(),
      platform::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(), dev_idx));
  return pool_[dev_idx]->New();
}

NpuEventResourcePool::NpuEventResourcePool() {
  int dev_cnt = platform::GetNPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetNPUDeviceId(dev_idx);
      aclrtEvent event;
      NPUEventCreate(&event);
      return event;
    };

    auto deleter = [dev_idx](aclrtEvent event) {
      platform::SetNPUDeviceId(dev_idx);
      NPUEventDestroy(event);
    };

    pool_.emplace_back(ResourcePool<NpuEventObject>::Create(creator, deleter));
  }
}

NpuEventResourcePool& NpuEventResourcePool::Instance() {
  static NpuEventResourcePool pool;
  return pool;
}

std::shared_ptr<NpuEventObject> NpuEventResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx, 0,
      platform::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx, pool_.size(),
      platform::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(), dev_idx));
  return pool_[dev_idx]->New();
}

}  // namespace platform
}  // namespace paddle

#endif
