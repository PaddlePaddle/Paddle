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

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/core/platform/device/custom/custom_device_resource_pool.h"

namespace paddle {
namespace platform {

CustomDeviceStreamResourcePool::CustomDeviceStreamResourcePool(
    const paddle::Place& place) {
  PADDLE_ENFORCE_EQ(
      phi::is_custom_place(place),
      true,
      common::errors::PreconditionNotMet(
          "Required device shall be CustomPlace, but received %d. ", place));

  int dev_cnt = phi::DeviceManager::GetDeviceCount(place.GetDeviceType());
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [place, dev_idx, this] {
      auto place_ = phi::CustomPlace(place.GetDeviceType(), dev_idx);
      phi::DeviceManager::SetDevice(place_);

      phi::stream::Stream* stream = new phi::stream::Stream;
      stream->Init(place_);
      this->streams_.push_back(stream);
      return stream;
    };

    pool_.emplace_back(ResourcePool<CustomDeviceStreamObject>::Create(
        creator, [](phi::stream::Stream* stream) {}));
  }
}

std::unordered_map<std::string, std::vector<CustomDeviceStreamResourcePool*>>&
CustomDeviceStreamResourcePool::GetMap() {
  static std::unordered_map<std::string,
                            std::vector<CustomDeviceStreamResourcePool*>>
      pool;
  return pool;
}

CustomDeviceStreamResourcePool::~CustomDeviceStreamResourcePool() {
  for (auto* p : streams_) {
    delete p;
  }
  pool_.clear();
}

void CustomDeviceStreamResourcePool::Release() {
  auto& pool = GetMap();
  for (auto& item : pool) {
    for (auto& p : item.second) {
      delete p;
    }
    item.second.clear();
  }
  pool.clear();
}

CustomDeviceStreamResourcePool& CustomDeviceStreamResourcePool::Instance(
    const paddle::Place& place) {
  auto& pool = GetMap();
  PADDLE_ENFORCE_EQ(
      phi::is_custom_place(place),
      true,
      common::errors::PreconditionNotMet(
          "Required device shall be CustomPlace, but received %d. ", place));
  if (pool.find(place.GetDeviceType()) == pool.end()) {
    pool.insert({place.GetDeviceType(),
                 std::vector<CustomDeviceStreamResourcePool*>()});
    for (size_t i = 0;
         i < phi::DeviceManager::GetDeviceCount(place.GetDeviceType());
         ++i) {
      pool[place.GetDeviceType()].emplace_back(
          new CustomDeviceStreamResourcePool(
              phi::CustomPlace(place.GetDeviceType(), i)));
    }
  }
  PADDLE_ENFORCE_LT(
      place.GetDeviceId(),
      pool[place.GetDeviceType()].size(),
      common::errors::OutOfRange("Device id is out of range, device id shall "
                                 "be less than %d, but received %d. ",
                                 pool[place.GetDeviceType()].size(),
                                 place.GetDeviceId()));
  return *pool[place.GetDeviceType()][place.GetDeviceId()];
}

std::shared_ptr<CustomDeviceStreamObject> CustomDeviceStreamResourcePool::New(
    int dev_idx) {
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

CustomDeviceEventResourcePool::CustomDeviceEventResourcePool(
    const paddle::Place& place) {
  PADDLE_ENFORCE_EQ(
      phi::is_custom_place(place),
      true,
      common::errors::PreconditionNotMet(
          "Required device shall be CustomPlace, but received %d. ", place));

  int dev_cnt = phi::DeviceManager::GetDeviceCount(place.GetDeviceType());
  pool_.reserve(dev_cnt);
  for (int dev_idx = 0; dev_idx < dev_cnt; ++dev_idx) {
    auto creator = [place, dev_idx, this] {
      auto place_ = phi::CustomPlace(place.GetDeviceType(), dev_idx);
      phi::DeviceManager::SetDevice(place_);

      phi::event::Event* event = new phi::event::Event;
      event->Init(place_);
      this->events_.push_back(event);
      return event;
    };

    pool_.emplace_back(ResourcePool<CustomDeviceEventObject>::Create(
        creator, [](phi::event::Event* event) {}));
  }
}

std::unordered_map<std::string, std::vector<CustomDeviceEventResourcePool*>>&
CustomDeviceEventResourcePool::GetMap() {
  static std::unordered_map<std::string,
                            std::vector<CustomDeviceEventResourcePool*>>
      pool;
  return pool;
}

CustomDeviceEventResourcePool::~CustomDeviceEventResourcePool() {
  for (auto* p : events_) {
    delete p;
  }
  pool_.clear();
}

void CustomDeviceEventResourcePool::Release() {
  auto& pool = GetMap();
  for (auto& item : pool) {
    for (auto& p : item.second) {
      delete p;
    }
    item.second.clear();
  }
  pool.clear();
}

CustomDeviceEventResourcePool& CustomDeviceEventResourcePool::Instance(
    const phi::Place& place) {
  auto& pool = GetMap();
  PADDLE_ENFORCE_EQ(
      phi::is_custom_place(place),
      true,
      common::errors::PreconditionNotMet(
          "Required device shall be CustomPlace, but received %d. ", place));
  if (pool.find(place.GetDeviceType()) == pool.end()) {
    pool.insert(
        {place.GetDeviceType(), std::vector<CustomDeviceEventResourcePool*>()});
    for (size_t i = 0;
         i < phi::DeviceManager::GetDeviceCount(place.GetDeviceType());
         ++i) {
      pool[place.GetDeviceType()].emplace_back(
          new CustomDeviceEventResourcePool(
              phi::CustomPlace(place.GetDeviceType(), i)));
    }
  }
  PADDLE_ENFORCE_LT(
      place.GetDeviceId(),
      pool[place.GetDeviceType()].size(),
      common::errors::OutOfRange("Device id is out of range, device id shall "
                                 "be less than %d, but received %d. ",
                                 pool[place.GetDeviceType()].size(),
                                 place.GetDeviceId()));
  return *pool[place.GetDeviceType()][place.GetDeviceId()];
}

std::shared_ptr<CustomDeviceEventObject> CustomDeviceEventResourcePool::New(
    int dev_idx) {
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
