/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include <limits>

#include "paddle/pten/api/ext/exception.h"

namespace paddle {
namespace experimental {

class Scalar {
 public:
  // Constructor support implicit
  Scalar(float val) : tag(Tag::HAS_F) { data_.f = val; }  // NOLINT

  Scalar(double val) : tag(Tag::HAS_D) { data_.d = val; }  // NOLINT

  Scalar(int32_t val) : tag(Tag::HAS_I32) { data_.i32 = val; }  // NOLINT

  Scalar(int64_t val) : tag(Tag::HAS_I64) { data_.i64 = val; }  // NOLINT

  Scalar(bool val) : tag(Tag::HAS_B) { data_.b = val; }  // NOLINT

  Scalar(const std::string& str_value) : tag(Tag::HAS_D) {  // NOLINT
    if (str_value == "inf") {
      data_.d = std::numeric_limits<double>::infinity();
    } else if (str_value == "-inf") {
      data_.d = -std::numeric_limits<double>::infinity();
    } else if (str_value == "nan") {
      data_.d = std::numeric_limits<double>::quiet_NaN();
    } else {
      data_.d = std::stod(str_value);
    }
  }

  template <typename T>
  inline T to() const {
    switch (tag) {
      case Tag::HAS_F:
        return static_cast<T>(data_.f);
      case Tag::HAS_D:
        return static_cast<T>(data_.d);
      case Tag::HAS_I32:
        return static_cast<T>(data_.i32);
      case Tag::HAS_I64:
        return static_cast<T>(data_.i64);
      case Tag::HAS_B:
        return static_cast<T>(data_.b);
      default:
        PD_THROW("Invalid enum scalar type tag `", static_cast<int>(tag), "`.");
    }
  }

 private:
  enum class Tag { HAS_F, HAS_D, HAS_I32, HAS_I64, HAS_B };
  Tag tag;

  union data {
    float f;
    double d;
    int32_t i32;
    int64_t i64;
    bool b;
  } data_;
};

}  // namespace experimental
}  // namespace paddle

namespace pten {
using Scalar = paddle::experimental::Scalar;
}  // namespace pten
