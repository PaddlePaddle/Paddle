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

namespace paddle {
namespace distributed {

template <typename KeyT, typename ValueT>
class GlobalMap final {
 public:
  static ValueT* Get(KeyT id) {
    ValueT* item = GetPPtr(id)->get();
    PADDLE_ENFORCE_NOT_NULL(
        item, platform::errors::NotFound("This value is not in global map."));
    return item;
  }
  template <typename... Args>
  static ValueT* Create(KeyT id, Args&&... args) {
    auto* ptr = GetPPtr(id);
    PADDLE_ENFORCE_EQ(ptr->get(), nullptr,
                      platform::errors::AlreadyExists(
                          "This value has already in global map."));
    ValueT* item = new ValueT(std::forward<Args>(args)...);
    ptr->reset(item);
    return item;
  }

 private:
  static std::unique_ptr<ValueT>* GetPPtr(KeyT id) {
    static std::mutex mutex;
    static std::unordered_map<KeyT, std::unique_ptr<ValueT>> id_to_ptr;
    std::unique_lock<std::mutex> lock(mutex);
    return &id_to_ptr[id];
  }
};
}  // namespace distributed
}  // namespace paddle
