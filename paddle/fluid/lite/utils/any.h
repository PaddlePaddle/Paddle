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
#include <functional>
#include <set>
#include "paddle/fluid/lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class Any {
 public:
  template <typename T>
  void set(const T& v) {
    set<T>();
    *get_mutable<T>() = v;
  }

  template <typename T>
  void set() {
    if (type_ != kInvalidType) {
      CHECK(type_ == typeid(T).hash_code());
    } else {
      type_ = typeid(T).hash_code();
      data_ = new T;
      deleter_ = [&] { delete static_cast<T*>(data_); };
    }
    data_ = new T;
  }

  template <typename T>
  const T& get() const {
    CHECK(data_);
    CHECK(type_ == typeid(T).hash_code());
    return *static_cast<T*>(data_);
  }
  template <typename T>
  T* get_mutable() {
    CHECK(data_);
    CHECK(type_ == typeid(T).hash_code());
    return static_cast<T*>(data_);
  }

  bool valid() const { return data_; }

 private:
  static size_t kInvalidType;
  size_t type_{kInvalidType};
  void* data_{};
  std::function<void()> deleter_;
};

}  // namespace lite
}  // namespace paddle
