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
#include <c10/core/Scalar.h>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

namespace at {

/// Extracts a scalar value from a single-element dense tensor.
/// Mirrors PyTorch's at::_local_scalar_dense: copies the tensor to CPU if
/// needed, then reads the first element according to its dtype.
inline at::Scalar _local_scalar_dense(const at::Tensor& self) {
  PD_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");

  // Move to CPU if necessary (for compatibility with PyTorch behavior)
  const PaddleTensor& inner = self._PD_GetInner();
  PaddleTensor cpu_tensor = inner;
  if (!phi::is_cpu_place(inner.place())) {
    PaddlePlace place(phi::AllocationType::CPU);
    cpu_tensor = inner.copy_to(place, /*blocking=*/true);
  }

  auto dtype = cpu_tensor.dtype();
  switch (dtype) {
    case phi::DataType::FLOAT32:
      return at::Scalar(*(cpu_tensor.data<float>()));
    case phi::DataType::FLOAT64:
      return at::Scalar(*(cpu_tensor.data<double>()));
    case phi::DataType::FLOAT16:
      return at::Scalar(
          static_cast<float>(*(cpu_tensor.data<phi::dtype::float16>())));
    case phi::DataType::BFLOAT16:
      return at::Scalar(
          static_cast<float>(*(cpu_tensor.data<phi::dtype::bfloat16>())));
    case phi::DataType::INT8:
      return at::Scalar(*(cpu_tensor.data<int8_t>()));
    case phi::DataType::INT16:
      return at::Scalar(*(cpu_tensor.data<int16_t>()));
    case phi::DataType::INT32:
      return at::Scalar(*(cpu_tensor.data<int32_t>()));
    case phi::DataType::INT64:
      return at::Scalar(*(cpu_tensor.data<int64_t>()));
    case phi::DataType::UINT8:
      return at::Scalar(*(cpu_tensor.data<uint8_t>()));
    case phi::DataType::BOOL:
      return at::Scalar(*(cpu_tensor.data<bool>()));
    case phi::DataType::COMPLEX64:
      return at::Scalar(*(cpu_tensor.data<phi::dtype::complex<float>>()));
    case phi::DataType::COMPLEX128:
      return at::Scalar(*(cpu_tensor.data<phi::dtype::complex<double>>()));
    default:
      PD_THROW("_local_scalar_dense: Unsupported data type");
  }
}

}  // namespace at
