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
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorOptions.h>
#include <utils/pinned_place.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

namespace at {

// Member function: Tensor::new_zeros
inline Tensor Tensor::new_zeros(at::IntArrayRef size,
                                at::TensorOptions options) const {
  auto actual_dtype =
      options.dtype_opt().has_value() ? options.dtype_opt().value() : dtype();
  auto actual_device = options.device_opt().has_value()
                           ? options.device_opt().value()
                           : device();
  auto actual_pin_memory = options.pinned_memory();

  auto pd_dtype = compat::_PD_AtenScalarTypeToPhiDataType(actual_dtype);
  auto pd_place = actual_device._PD_GetInner();

  paddle::Tensor result;
  if (actual_pin_memory) {
    // Pinning memory is only supported for CPU tensors
    if (options.has_device() && !actual_device.is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(pd_place);
    auto dense_cpu = paddle::experimental::zeros(
        size._PD_ToPaddleIntArray(), pd_dtype, phi::CPUPlace());
    result = dense_cpu.copy_to(pinned_place, /*blocking=*/true);
  } else {
    result = paddle::experimental::zeros(
        size._PD_ToPaddleIntArray(), pd_dtype, pd_place);
  }
  return Tensor(result);
}

inline Tensor Tensor::new_zeros(at::IntArrayRef size,
                                ::std::optional<at::ScalarType> dtype,
                                ::std::optional<at::Layout>,
                                ::std::optional<at::Device> device,
                                ::std::optional<bool> pin_memory) const {
  auto options = at::TensorOptions()
                     .dtype(dtype.value_or(this->dtype()))
                     .device(device.value_or(this->device()))
                     .pinned_memory(pin_memory);
  return new_zeros(size, options);
}

}  // namespace at
