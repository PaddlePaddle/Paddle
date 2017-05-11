/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <paddle/utils/Any.h>
#include <string>
#include <unordered_map>

namespace paddle {
namespace topology {

class Attribute : public std::unordered_map<std::string, any> {
public:
  template <typename T>
  const T& get(const std::string& name) const {
    auto attrPtr = &at(name);
    auto* ptr = any_cast<T>(attrPtr);
    if (ptr == nullptr) throw bad_any_cast();
    return *ptr;
  }

  template <typename T>
  T& get(const std::string& name) {
    auto attrPtr = &at(name);
    auto* ptr = any_cast<T>(attrPtr);
    if (ptr == nullptr) throw bad_any_cast();
    return *ptr;
  }
};

}  // namespace topology
}  // namespace paddle
