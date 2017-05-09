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

#include <string>
#include <unordered_map>
#include "paddle/utils/Any.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace function {

/**
 * Function Configuration.
 * The argument type of Function::init.
 */
class Config {
public:
  template <typename T>
  T get(const std::string& key, Error* err = nullptr) const {
    try {
      return any_cast<T>(valueMap_.at(key));
    } catch (std::exception& e) {  // could be cast or out of range exception.
      if (err) {
        *err = Error(e.what());
      } else {
        LOG(FATAL) << "Cannot get key " << key << " with error " << e.what();
      }
      return T();
    }
  }

  template <typename T>
  Config& set(const std::string& key, T v, Error* err = nullptr) {
    auto it = valueMap_.find(key);
    if (it != valueMap_.end()) {  // already contains key.
      if (err) {
        *err = Error("Key %s is already set in FuncConfig", key.c_str());
      } else {
        LOG(FATAL) << "Key " << key << " is already set in FuncConfig.";
      }
      return *this;
    }
    valueMap_[key] = any(v);
    return *this;
  }

protected:
  mutable std::unordered_map<std::string, any> valueMap_;
};
}  // namespace function
}  // namespace paddle
