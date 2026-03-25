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
  // Paddle's allclose returns a Tensor, but PyTorch's allclose returns bool.
  // The allclose kernel always sets output dtype to phi::DataType::BOOL via
  // kernel->OutputAt(0).SetDataType(phi::DataType::BOOL), so we read BOOL
  // directly.
  PaddleTensor result = paddle::experimental::allclose(self._PD_GetInner(),
                                                       other._PD_GetInner(),
                                                       phi::Scalar(rtol),
                                                       phi::Scalar(atol),
                                                       equal_nan);
  auto* result_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(result.impl()).get();
  return *result_tensor->data<bool>();
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
