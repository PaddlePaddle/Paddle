/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/common/scalar.h"

#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace experimental {

void ThrowErrorIf(int num) {
  PADDLE_ENFORCE_EQ(num,
                    1,
                    phi::errors::InvalidArgument(
                        "The Scalar only supports Tensor with 1 element, but "
                        "now Tensor has `%d` elements",
                        num));
}
// The Tensor must have one dim
/*
template <typename T>
ScalarBase<T>::ScalarBase(const T& tensor) : dtype_(tensor.dtype()) {  // NOLINT
    is_from_tensor_ = true;
    PADDLE_ENFORCE_EQ(tensor.numel(),
                      1,
                      phi::errors::InvalidArgument(
                          "The Scalar only supports Tensor with 1 element, but "
                          "now Tensor has `%d` elements",
                          tensor.numel()));

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
  */
}  // namespace experimental
}  // namespace paddle
