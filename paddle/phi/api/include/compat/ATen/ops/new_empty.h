// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <c10/core/TensorOptions.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

// Member function: Tensor::new_empty
inline Tensor Tensor::new_empty(at::IntArrayRef size,
                                at::TensorOptions options) const {
  auto actual_dtype =
      options.dtype_opt().has_value() ? options.dtype_opt().value() : dtype();
  auto actual_device = options.device_opt().has_value()
                           ? options.device_opt().value()
                           : device();

  auto pd_dtype = compat::_PD_AtenScalarTypeToPhiDataType(actual_dtype);
  auto pd_place = actual_device._PD_GetInner();

  auto result = paddle::experimental::empty(
      size._PD_ToPaddleIntArray(), pd_dtype, pd_place);
  return Tensor(result);
}

inline Tensor Tensor::new_empty(at::IntArrayRef size,
                                ::std::optional<at::ScalarType> dtype,
                                ::std::optional<at::Layout>,
                                ::std::optional<at::Device> device,
                                ::std::optional<bool>) const {
  auto actual_dtype = dtype.has_value() ? dtype.value() : this->dtype();
  auto actual_device = device.has_value() ? device.value() : this->device();

  auto pd_dtype = compat::_PD_AtenScalarTypeToPhiDataType(actual_dtype);
  auto pd_place = actual_device._PD_GetInner();

  auto result = paddle::experimental::empty(
      size._PD_ToPaddleIntArray(), pd_dtype, pd_place);
  return Tensor(result);
}

}  // namespace at
