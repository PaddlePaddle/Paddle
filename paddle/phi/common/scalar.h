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

#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/api/include/tensor.h"

namespace paddle {
namespace experimental {

void ThrowTensorConvertError(int);

template <typename T>
class ScalarBase {
 public:
  // Constructor support implicit
  ScalarBase(double val) : dtype_(DataType::FLOAT64) {  // NOLINT
    data_.f64 = val;
  }

  ScalarBase(float val) : dtype_(DataType::FLOAT32) {  // NOLINT
    data_.f32 = val;
  }

  ScalarBase(float16 val) : dtype_(DataType::FLOAT16) {  // NOLINT
    data_.f16 = val;
  }

  ScalarBase(bfloat16 val) : dtype_(DataType::BFLOAT16) {  // NOLINT
    data_.bf16 = val;
  }

  ScalarBase(int64_t val) : dtype_(DataType::INT64) {  // NOLINT
    data_.i64 = val;
  }

  ScalarBase(int32_t val) : dtype_(DataType::INT32) {  // NOLINT
    data_.i32 = val;
  }

  ScalarBase(int16_t val) : dtype_(DataType::INT16) {  // NOLINT
    data_.i16 = val;
  }

  ScalarBase(int8_t val) : dtype_(DataType::INT8) {  // NOLINT
    data_.i8 = val;
  }

  ScalarBase(uint64_t val) : dtype_(DataType::UINT64) {  // NOLINT
    data_.ui64 = val;
  }

  ScalarBase(uint32_t val) : dtype_(DataType::UINT32) {  // NOLINT
    data_.ui32 = val;
  }

  ScalarBase(uint16_t val) : dtype_(DataType::UINT16) {  // NOLINT
    data_.ui16 = val;
  }

  ScalarBase(uint8_t val) : dtype_(DataType::UINT8) {  // NOLINT
    data_.ui8 = val;
  }

  ScalarBase(bool val) : dtype_(DataType::BOOL) {  // NOLINT
    data_.b = val;
  }

  ScalarBase(complex64 val) : dtype_(DataType::COMPLEX64) {  // NOLINT
    data_.c64 = val;
  }

  ScalarBase(complex128 val) : dtype_(DataType::COMPLEX128) {  // NOLINT
    data_.c128 = val;
  }

  // The compatible method for fliud operators,
  // and it will be removed in the future.
  explicit ScalarBase(const std::string& str_value)
      : dtype_(DataType::FLOAT64) {
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

  // The Tensor must have one dim
  ScalarBase(const T& tensor) : dtype_(tensor.dtype()) {  // NOLINT
    is_from_tensor_ = true;
    ThrowTensorConvertError(tensor.numel());
    switch (dtype_) {
      case DataType::FLOAT32:
        data_.f32 = tensor.template data<float>()[0];
        break;
      case DataType::FLOAT64:
        data_.f64 = tensor.template data<double>()[0];
        break;
      case DataType::FLOAT16:
        data_.f16 = tensor.template data<float16>()[0];
        break;
      case DataType::BFLOAT16:
        data_.bf16 = tensor.template data<bfloat16>()[0];
        break;
      case DataType::INT32:
        data_.i32 = tensor.template data<int32_t>()[0];
        break;
      case DataType::INT64:
        data_.i64 = tensor.template data<int64_t>()[0];
        break;
      case DataType::INT16:
        data_.i16 = tensor.template data<int16_t>()[0];
        break;
      case DataType::INT8:
        data_.i8 = tensor.template data<int8_t>()[0];
        break;
      case DataType::UINT8:
        data_.ui8 = tensor.template data<uint8_t>()[0];
        break;
      case DataType::BOOL:
        data_.b = tensor.template data<bool>()[0];
        break;
      case DataType::COMPLEX64:
        data_.c64 = tensor.template data<complex64>()[0];
        break;
      case DataType::COMPLEX128:
        data_.c128 = tensor.template data<complex128>()[0];
        break;
      default:
        PD_THROW("Invalid tensor data type `", dtype_, "`.");
    }
  }

  template <typename OtherT>
  ScalarBase(const ScalarBase<OtherT>& other) {
    CopyScalar(other, this);
  }

  // NOTE(xiongkun): some op need to judge the dtype of the Scalar, we expose a
  // interface.
  bool FromTensor() const { return is_from_tensor_; }

  void SetFromTensor(bool from_tensor) { is_from_tensor_ = from_tensor; }

  template <typename RT>
  inline RT to() const {
    switch (dtype_) {
      case DataType::FLOAT32:
        return static_cast<RT>(data_.f32);
      case DataType::FLOAT64:
        return static_cast<RT>(data_.f64);
      case DataType::FLOAT16:
        return static_cast<RT>(data_.f16);
      case DataType::BFLOAT16:
        return static_cast<RT>(data_.bf16);
      case DataType::INT32:
        return static_cast<RT>(data_.i32);
      case DataType::INT64:
        return static_cast<RT>(data_.i64);
      case DataType::INT16:
        return static_cast<RT>(data_.i16);
      case DataType::INT8:
        return static_cast<RT>(data_.i8);
      case DataType::UINT16:
        return static_cast<RT>(data_.ui16);
      case DataType::UINT8:
        return static_cast<RT>(data_.ui8);
      case DataType::BOOL:
        return static_cast<RT>(data_.b);
      case DataType::COMPLEX64:
        return static_cast<RT>(data_.c64);
      case DataType::COMPLEX128:
        return static_cast<RT>(data_.c128);
      default:
        PD_THROW("Invalid enum scalar data type `", dtype_, "`.");
    }
  }

  DataType dtype() const { return dtype_; }

 private:
  template <typename T1, typename T2>
  friend void CopyScalar(const ScalarBase<T1>& src, ScalarBase<T2>* dst);

 private:
  bool is_from_tensor_{false};
  DataType dtype_;
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

template <typename T1, typename T2>
void CopyScalar(const ScalarBase<T1>& src, ScalarBase<T2>* dst) {
  dst->dtype_ = src.dtype_;
  dst->data_.c128 = src.data_.c128;
}

using Scalar = paddle::experimental::ScalarBase<paddle::experimental::Tensor>;

}  // namespace experimental
}  // namespace paddle

namespace phi {
class DenseTensor;
using Scalar = paddle::experimental::ScalarBase<DenseTensor>;
}  // namespace phi
