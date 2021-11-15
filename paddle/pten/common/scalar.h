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

#include "paddle/fluid/platform/enforce.h"
#include "paddle/pten/common/data_type.h"

namespace paddle {
namespace experimental {

class Scalar {
 public:
  // Constructor support implicit
  Scalar(double val) : data_type_(DataType::FLOAT64) {  // NOLINT
    data_.f64 = val;
  }

  Scalar(float val) : data_type_(DataType::FLOAT32) {  // NOLINT
    data_.f32 = val;
  }

  Scalar(float16 val) : data_type_(DataType::FLOAT16) {  // NOLINT
    data_.f16 = val;
  }

  Scalar(bfloat16 val) : data_type_(DataType::BFLOAT16) {  // NOLINT
    data_.bf16 = val;
  }

  Scalar(int64_t val) : data_type_(DataType::INT64) {  // NOLINT
    data_.i64 = val;
  }

  Scalar(int32_t val) : data_type_(DataType::INT32) {  // NOLINT
    data_.i32 = val;
  }

  Scalar(int16_t val) : data_type_(DataType::INT16) {  // NOLINT
    data_.i16 = val;
  }

  Scalar(int8_t val) : data_type_(DataType::INT8) { data_.i8 = val; }  // NOLINT

  Scalar(uint64_t val) : data_type_(DataType::UINT64) {  // NOLINT
    data_.ui64 = val;
  }

  Scalar(uint32_t val) : data_type_(DataType::UINT32) {  // NOLINT
    data_.ui32 = val;
  }

  Scalar(uint16_t val) : data_type_(DataType::UINT16) {  // NOLINT
    data_.ui16 = val;
  }

  Scalar(uint8_t val) : data_type_(DataType::UINT8) {  // NOLINT
    data_.ui8 = val;
  }

  Scalar(bool val) : data_type_(DataType::BOOL) { data_.b = val; }  // NOLINT

  Scalar(complex64 val) : data_type_(DataType::COMPLEX64) {  // NOLINT
    data_.c64 = val;
  }

  Scalar(complex128 val) : data_type_(DataType::COMPLEX128) {  // NOLINT
    data_.c128 = val;
  }

  Scalar(const std::string& str_value)  // NOLINT
      : data_type_(DataType::FLOAT64) {
    if (str_value == "inf") {
      data_.f64 = std::numeric_limits<double>::infinity();
    } else if (str_value == "-inf") {
      data_.f64 = -std::numeric_limits<double>::infinity();
    } else if (str_value == "nan") {
      data_.f64 = std::numeric_limits<double>::quiet_NaN();
    } else {
      data_.f64 = std::stod(str_value);
    }
  }

  template <typename T>
  inline T to() const {
    switch (data_type_) {
      case DataType::FLOAT32:
        return static_cast<T>(data_.f32);
      case DataType::FLOAT64:
        return static_cast<T>(data_.f64);
      case DataType::FLOAT16:
        return static_cast<T>(data_.f16);
      case DataType::BFLOAT16:
        return static_cast<T>(data_.bf16);
      case DataType::INT32:
        return static_cast<T>(data_.i32);
      case DataType::INT64:
        return static_cast<T>(data_.i64);
      case DataType::INT16:
        return static_cast<T>(data_.i16);
      case DataType::INT8:
        return static_cast<T>(data_.i8);
      case DataType::UINT32:
        return static_cast<T>(data_.ui32);
      case DataType::UINT64:
        return static_cast<T>(data_.ui64);
      case DataType::UINT16:
        return static_cast<T>(data_.ui16);
      case DataType::UINT8:
        return static_cast<T>(data_.ui8);
      case DataType::BOOL:
        return static_cast<T>(data_.b);
      case DataType::COMPLEX64:
        return static_cast<T>(data_.c64);
      case DataType::COMPLEX128:
        return static_cast<T>(data_.c128);
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid enum scalar type tag `%s`.", data_type_));
    }
  }

 private:
  DataType data_type_;
  union data {
    bool b;
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t ui8;
    uint16_t ui16;
    uint32_t ui32;
    uint64_t ui64;
    bfloat16 bf16;
    float16 f16;
    float f32;
    double f64;
    complex64 c64;
    complex128 c128;
  } data_;
};

}  // namespace experimental
}  // namespace paddle

namespace pten {
using Scalar = paddle::experimental::Scalar;
}
