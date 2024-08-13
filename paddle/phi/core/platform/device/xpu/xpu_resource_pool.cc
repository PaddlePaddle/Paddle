// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_XPU)
#include "paddle/phi/core/platform/device/xpu/xpu_resource_pool.h"

namespace paddle {
namespace platform {

XpuStreamResourcePool::XpuStreamResourcePool() {
  int dev_cnt = platform::GetXPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      phi::backends::xpu::XPUDeviceGuard guard(dev_idx);
      xpuStream stream;
      xpu_stream_create(&stream);
      return stream;
    };

    auto deleter = [dev_idx](xpuStream stream) {
      phi::backends::xpu::XPUDeviceGuard guard(dev_idx);
      xpu_stream_destroy(stream);
    };

    pool_.emplace_back(ResourcePool<XpuStreamObject>::Create(creator, deleter));
  }
}

XpuStreamResourcePool& XpuStreamResourcePool::Instance() {
  static XpuStreamResourcePool pool;
  return pool;
}

std::shared_ptr<XpuStreamObject> XpuStreamResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx,
      0,
      common::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx,
      pool_.size(),
      common::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(),
          dev_idx));
  return pool_[dev_idx]->New();
}

XpuEventResourcePool::XpuEventResourcePool() {
  int dev_cnt = platform::GetXPUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      phi::backends::xpu::XPUDeviceGuard guard(dev_idx);
      xpuEventHandle event;
      xpu_event_create(&event);
      return event;
    };

    auto deleter = [dev_idx](xpuEventHandle event) {
      phi::backends::xpu::XPUDeviceGuard guard(dev_idx);
      xpu_event_destroy(event);
    };

    pool_.emplace_back(ResourcePool<XpuEventObject>::Create(creator, deleter));
  }
}

XpuEventResourcePool& XpuEventResourcePool::Instance() {
  static XpuEventResourcePool pool;
  return pool;
}

std::shared_ptr<XpuEventObject> XpuEventResourcePool::New(int dev_idx) {
  PADDLE_ENFORCE_GE(
      dev_idx,
      0,
      common::errors::InvalidArgument(
          "The dev_idx should be not less than 0, but got %d.", dev_idx));
  PADDLE_ENFORCE_LT(
      dev_idx,
      pool_.size(),
      common::errors::OutOfRange(
          "The dev_idx should be less than device count %d, but got %d.",
          pool_.size(),
          dev_idx));
  return pool_[dev_idx]->New();
}

}  // namespace platform
}  // namespace paddle
#endif
