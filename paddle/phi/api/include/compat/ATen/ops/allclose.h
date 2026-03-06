// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <ATen/core/Tensor.h>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"

namespace at {

// allclose: Check if two tensors are close to each other
inline bool allclose(const at::Tensor& self,
                     const at::Tensor& other,
                     double rtol = 1e-05,
                     double atol = 1e-08,
                     bool equal_nan = false) {
  // Paddle's allclose returns a Tensor, but PyTorch's allclose returns bool
  // We need to extract the scalar value from the result tensor
  // Use phi::Scalar instead of paddle::experimental::Scalar to ensure
  // correct dtype is passed to the kernel
  PaddleTensor result = paddle::experimental::allclose(self._PD_GetInner(),
                                                       other._PD_GetInner(),
                                                       phi::Scalar(rtol),
                                                       phi::Scalar(atol),
                                                       equal_nan);

  // Extract the boolean value from the result tensor
  // allclose should return a scalar tensor with a single boolean value
  auto* result_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(result.impl()).get();
  if (!result_tensor || result_tensor->numel() != 1) {
    PD_THROW("allclose: expected scalar tensor result");
  }

  // Read the value from the tensor (could be bool, int8, int32, etc.)
  auto dtype = result_tensor->dtype();
  if (dtype == phi::DataType::BOOL) {
    bool* bool_ptr = result_tensor->data<bool>();
    return *bool_ptr;
  } else if (dtype == phi::DataType::INT8) {
    int8_t* int8_ptr = result_tensor->data<int8_t>();
    return *int8_ptr != 0;
  } else if (dtype == phi::DataType::INT32) {
    int32_t* int32_ptr = result_tensor->data<int32_t>();
    return *int32_ptr != 0;
  } else if (dtype == phi::DataType::INT64) {
    int64_t* int64_ptr = result_tensor->data<int64_t>();
    return *int64_ptr != 0;
  } else {
    PD_THROW("allclose: unsupported result dtype");
  }
}

}  // namespace at

namespace at {

// Tensor member function implementation
inline bool Tensor::allclose(const at::Tensor& other,
                             double rtol,
                             double atol,
                             bool equal_nan) const {
  return at::allclose(*this, other, rtol, atol, equal_nan);
}

}  // namespace at
