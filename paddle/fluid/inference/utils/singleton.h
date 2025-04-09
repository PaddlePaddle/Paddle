/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <unordered_map>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {

// NOTE not thread-safe.
template <typename T>
struct Singleton {
  static T& Global() {
    static T* x = new T;
    return *x;
  }

  Singleton() = delete;
  Singleton& operator=(const Singleton&) = delete;
};

/*
 * An Registry for any type.
 * NOTE not thread-safe.
 */
template <typename ItemParent>
struct Registry {
  static Registry& Global() {
    static auto* x = new Registry<ItemParent>;
    return *x;
  }

  template <typename ItemChild>
  void Register(const std::string& name) {
    PADDLE_ENFORCE_EQ(
        items_.count(name),
        0,
        common::errors::AlreadyExists("Item `%s` has been registered.", name));
    items_[name] = new ItemChild;
  }

  ItemParent* Lookup(const std::string& name,
                     const std::string& default_name = "") {
    auto it = items_.find(name);
    if (it == items_.end()) {
      if (default_name == "")
        return nullptr;
      else
        return items_.find(default_name)->second;
    }
    return it->second;
  }

  ~Registry() {
    for (auto& item : items_) {
      delete item.second;
    }
  }

 private:
  Registry() = default;
  std::unordered_map<std::string, ItemParent*> items_;
};

}  // namespace inference
}  // namespace paddle
