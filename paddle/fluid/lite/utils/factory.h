// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include "paddle/fluid/lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

/*
 * Factor for any Type creator.
 *
 * Usage:
 *
 * struct SomeType;
 * // Register a creator.
 * Factory<SomeType>::Global().Register("some_key", [] ->
 *                                      std::unique_ptr<SomeType> { ... });
 * // Retrive a creator.
 * auto some_type_instance = Factory<SomeType>::Global().Create("some_key");
 */
template <typename ItemType, typename ItemTypePtr>
class Factory {
 public:
  using item_t = ItemType;
  using self_t = Factory<item_t, ItemTypePtr>;
  using item_ptr_t = ItemTypePtr;
  using creator_t = std::function<item_ptr_t()>;

  static Factory& Global() {
    static Factory* x = new self_t;
    return *x;
  }

  void Register(const std::string& op_type, creator_t&& creator) {
    creators_[op_type].emplace_back(std::move(creator));
  }

  item_ptr_t Create(const std::string& op_type) const {
    auto res = Creates(op_type);
    if (res.empty()) return nullptr;
    CHECK_EQ(res.size(), 1UL) << "Get multiple Op for type " << op_type;
    return std::move(res.front());
  }

  std::list<item_ptr_t> Creates(const std::string& op_type) const {
    std::list<item_ptr_t> res;
    auto it = creators_.find(op_type);
    if (it == creators_.end()) return res;
    for (auto& c : it->second) {
      res.emplace_back(c());
    }
    return res;
  }

  std::string DebugString() const {
    std::stringstream ss;
    for (const auto& item : creators_) {
      ss << "  - " << item.first << std::endl;
    }
    return ss.str();
  }

 protected:
  std::unordered_map<std::string, std::list<creator_t>> creators_;
};

/* A helper function to help run a lambda at the start.
 */
template <typename Type>
class Registor {
 public:
  explicit Registor(std::function<void()>&& functor) { functor(); }

  // Touch will do nothing.
  int Touch() { return 0; }
};

}  // namespace lite
}  // namespace paddle
