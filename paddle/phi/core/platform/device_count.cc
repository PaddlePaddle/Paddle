// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/device_count.h"

#include <mutex>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

static std::mutex& GetRegistryMutex() {
  static std::mutex mu;
  return mu;
}

static auto& GetRegistry() {
  static std::unordered_map<std::string, DeviceCountFn> registry;
  return registry;
}

void RegisterDeviceCountProvider(const std::string& device_type,
                                 DeviceCountFn fn) {
  PADDLE_ENFORCE_NOT_NULL(
      fn,
      common::errors::InvalidArgument(
          "DeviceCountFn must not be null for device_type '%s'.", device_type));
  std::lock_guard<std::mutex> lock(GetRegistryMutex());
  PADDLE_ENFORCE_EQ(GetRegistry().count(device_type),
                    0,
                    common::errors::AlreadyExists(
                        "DeviceCountProvider for '%s' already registered. "
                        "Duplicate registration is not allowed.",
                        device_type));
  GetRegistry()[device_type] = fn;
}

std::optional<int> GetDeviceCount(const std::string& device_type) {
  std::lock_guard<std::mutex> lock(GetRegistryMutex());
  auto& reg = GetRegistry();
  auto it = reg.find(device_type);
  if (it == reg.end()) {
    LOG(WARNING) << "DeviceCountProvider for '" << device_type
                 << "' not registered, returning nullopt.";
    return std::nullopt;
  }
  return it->second();
}

}  // namespace phi
