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

#pragma once

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include <memory>
#include <type_traits>
#include <vector>

#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/resource_pool.h"

namespace paddle {
namespace platform {

using CustomDeviceStreamObject = phi::stream::Stream;
using CustomDeviceEventObject = phi::event::Event;

class CustomDeviceStreamResourcePool {
 public:
  static std::unordered_map<std::string,
                            std::vector<CustomDeviceStreamResourcePool*>>&
  GetMap();

  static void Release();

  std::shared_ptr<CustomDeviceStreamObject> New(int dev_idx);

  static CustomDeviceStreamResourcePool& Instance(const paddle::Place& place);

  ~CustomDeviceStreamResourcePool();

 private:
  explicit CustomDeviceStreamResourcePool(const paddle::Place& place);

  DISABLE_COPY_AND_ASSIGN(CustomDeviceStreamResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<CustomDeviceStreamObject>>> pool_;
  std::vector<phi::stream::Stream*> streams_;
};

class CustomDeviceEventResourcePool {
 public:
  std::shared_ptr<CustomDeviceEventObject> New(int dev_idx);

  static std::unordered_map<std::string,
                            std::vector<CustomDeviceEventResourcePool*>>&
  GetMap();

  static void Release();

  static CustomDeviceEventResourcePool& Instance(const paddle::Place& place);

  ~CustomDeviceEventResourcePool();

 private:
  explicit CustomDeviceEventResourcePool(const paddle::Place& place);

  DISABLE_COPY_AND_ASSIGN(CustomDeviceEventResourcePool);

 private:
  std::vector<std::shared_ptr<ResourcePool<CustomDeviceEventObject>>> pool_;
  std::vector<phi::event::Event*> events_;
};

}  // namespace platform
}  // namespace paddle

#endif
