// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>
#include <absl/types/any.h>

#include <string>

namespace cinn {
namespace common {

/**
 * Key value.
 */
class InfoRegistry {
 public:
  template <typename T>
  T& Get(const std::string& key);

  size_t size() const { return data_.size(); }

  void Clear() { data_.clear(); }

 private:
  absl::flat_hash_map<std::string, absl::any> data_;
};

template <typename T>
T& InfoRegistry::Get(const std::string& key) {
  auto it = data_.find(key);
  if (it == data_.end()) {
    data_[key] = T();
  }
  return absl::any_cast<T&>(data_[key]);
}

}  // namespace common
}  // namespace cinn
