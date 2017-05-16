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
#include "TypeDefs.h"
#include "paddle/utils/Any.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace topology {
namespace meta {

class AttributeMap : public Map<std::string, any> {
public:
  template <typename T>
  Error __must_check set(const std::string& name,
                         const T& val,
                         bool overwrite = false) {
    if (!overwrite && find(name) != end()) {
      return Error("Attribute %s has been set", name.c_str());
    }
    (*this)[name] = val;
    return Error();
  }

  template <typename T>
  Error __must_check get(const std::string& name, const T** ptr) const {
    auto it = find(name);
    if (it == end())
      return Error("Attribute %s has not been set", name.c_str());
    auto* valPtr = &it->second;
    *ptr = any_cast<T>(valPtr);
    if (*ptr == nullptr) {
      return Error("Attibute %s type mismatch, expect %s, actual %s",
                   name.c_str(),
                   typeid(T).name(),
                   valPtr->type().name());
    }
    return Error();
  }

  template <typename T>
  Error __must_check get(const std::string& name, T** ptr) {
    const T* tmp;
    auto err = this->get<T>(name, &tmp);
    if (!err.isOK()) {
      return err;
    }
    *ptr = const_cast<T*>(tmp);
    return Error();
  }

  template <typename T>
  const T& get(const std::string& name) const {
    const T* ptr;
    this->get<T>(name, &ptr).check();
    return *ptr;
  }

  template <typename T>
  T& get(const std::string& name) {
    T* ptr;
    this->get<T>(name, &ptr).check();
    return *ptr;
  }

  template <typename T>
  T get(const std::string& name, T defaultVal) const {
    const T* tmp;
    auto err = get(name, &tmp);
    if (err.isOK()) {
      return *tmp;
    } else {
      return defaultVal;
    }
  }
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
