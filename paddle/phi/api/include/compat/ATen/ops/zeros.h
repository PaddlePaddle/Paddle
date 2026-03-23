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
#include <utils/dense_sparse_conversion.h>
#include <utils/pinned_place.h>

#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

namespace at {

inline at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options = {}) {
  if (options.pinned_memory()) {
    // Pinning memory is only supported for CPU tensors
    if (options.has_device() && !options.device().is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(base_place);
    auto dense = paddle::experimental::zeros(
        size._PD_ToPaddleIntArray(),
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
        phi::CPUPlace());
    dense = dense.copy_to(pinned_place, /*blocking=*/true);
    return compat::_PD_ConvertToSparseIfNeeded(dense, options.layout());
  }
  auto dense = paddle::experimental::zeros(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
  return compat::_PD_ConvertToSparseIfNeeded(dense, options.layout());
}

inline at::Tensor zeros(at::IntArrayRef size,
                        ::std::optional<at::ScalarType> dtype,
                        ::std::optional<at::Layout> layout,
                        ::std::optional<at::Device> device,
                        ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype.value_or(c10::get_default_dtype()))
                     .layout(layout)
                     .device(device.value_or(at::kCPU))
                     .pinned_memory(pin_memory);
  return zeros(size, options);
}

inline at::Tensor zeros_symint(c10::SymIntArrayRef size,
                               at::TensorOptions options = {}) {
  if (options.pinned_memory()) {
    // Pinning memory is only supported for CPU tensors
    if (options.has_device() && !options.device().is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(base_place);
    auto dense = paddle::experimental::zeros(
        size._PD_ToPaddleIntArray(),
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
        phi::CPUPlace());
    dense = dense.copy_to(pinned_place, /*blocking=*/true);
    return compat::_PD_ConvertToSparseIfNeeded(dense, options.layout());
  }
  auto dense = paddle::experimental::zeros(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
  return compat::_PD_ConvertToSparseIfNeeded(dense, options.layout());
}

inline at::Tensor zeros_symint(c10::SymIntArrayRef size,
                               ::std::optional<at::ScalarType> dtype,
                               ::std::optional<at::Layout> layout,
                               ::std::optional<at::Device> device,
                               ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype.value_or(c10::get_default_dtype()))
                     .layout(layout)
                     .device(device.value_or(at::kCPU))
                     .pinned_memory(pin_memory);
  return zeros_symint(size, options);
}

}  // namespace at
