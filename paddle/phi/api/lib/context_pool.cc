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

#include "paddle/phi/api/include/context_pool.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace experimental {

DeviceContextPool& DeviceContextPool::Instance() {
  static DeviceContextPool g_device_context_pool;
  return g_device_context_pool;
}

const phi::DeviceContext* DeviceContextPool::Get(const Place& place) {
  auto it = context_map_.find(place);
  if (it == context_map_.end()) {
    // only when we need the specific DeviceContext, get and cache it
    auto* dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      context_map_[place] = dev_ctx;
    }
    return dev_ctx;
  }
  return it->second;
}

phi::DeviceContext* DeviceContextPool::GetMutable(const Place& place) {
  return const_cast<phi::DeviceContext*>(Get(place));
}

}  // namespace experimental
}  // namespace paddle
