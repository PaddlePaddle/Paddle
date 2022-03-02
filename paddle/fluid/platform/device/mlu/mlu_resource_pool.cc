// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_MLU)
#include "paddle/fluid/platform/device/mlu/mlu_resource_pool.h"

namespace paddle {
namespace platform {

MluStreamResourcePool::MluStreamResourcePool() {
  int dev_cnt = platform::GetMLUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetMLUDeviceId(dev_idx);
      mluStream stream;
      cnrtQueueCreate(&stream);
      return stream;
    };

    auto deleter = [dev_idx](mluStream stream) {
      platform::SetMLUDeviceId(dev_idx);
      cnrtQueueDestroy(stream);
    };

    pool_.emplace_back(ResourcePool<MluStreamObject>::Create(creator, deleter));
  }
}

MluStreamResourcePool& MluStreamResourcePool::Instance() {
  static MluStreamResourcePool pool;
  return pool;
}

std::shared_ptr<MluStreamObject> MluStreamResourcePool::New(int dev_idx) {
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

MluEventResourcePool::MluEventResourcePool() {
  int dev_cnt = platform::GetMLUDeviceCount();
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [dev_idx] {
      platform::SetMLUDeviceId(dev_idx);
      mluEventHandle event;
      cnrtNotifierCreate(&event);
      return event;
    };

    auto deleter = [dev_idx](mluEventHandle event) {
      platform::SetMLUDeviceId(dev_idx);
      cnrtNotifierDestroy(event);
    };

    pool_.emplace_back(ResourcePool<MluEventObject>::Create(creator, deleter));
  }
}

MluEventResourcePool& MluEventResourcePool::Instance() {
  static MluEventResourcePool pool;
  return pool;
}

std::shared_ptr<MluEventObject> MluEventResourcePool::New(int dev_idx) {
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
