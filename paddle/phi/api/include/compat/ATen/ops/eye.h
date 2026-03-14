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
#include <c10/core/TensorOptions.h>
#include <optional>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

namespace at {

// eye(n) — n×n identity matrix
inline at::Tensor eye(int64_t n, at::TensorOptions options = {}) {
  if (options.pinned_memory()) {
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = phi::is_xpu_place(base_place)
                                  ? phi::Place(phi::XPUPinnedPlace())
                                  : phi::Place(phi::GPUPinnedPlace());
    auto dense = paddle::experimental::eye(
        n,
        /*num_columns=*/-1,
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
        phi::CPUPlace());
    return dense.copy_to(pinned_place, /*blocking=*/true);
  }
  return paddle::experimental::eye(
      n,
      /*num_columns=*/-1,
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
}

// eye(n, m) — n×m identity-like matrix
inline at::Tensor eye(int64_t n, int64_t m, at::TensorOptions options = {}) {
  if (options.pinned_memory()) {
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = phi::is_xpu_place(base_place)
                                  ? phi::Place(phi::XPUPinnedPlace())
                                  : phi::Place(phi::GPUPinnedPlace());
    auto dense = paddle::experimental::eye(
        n,
        m,
        compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
        phi::CPUPlace());
    return dense.copy_to(pinned_place, /*blocking=*/true);
  }
  return paddle::experimental::eye(
      n,
      m,
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
}

// eye(n, dtype, layout, device, pin_memory)
inline at::Tensor eye(int64_t n,
                      ::std::optional<at::ScalarType> dtype,
                      ::std::optional<at::Layout> layout,
                      ::std::optional<at::Device> device,
                      ::std::optional<bool> pin_memory) {
  PD_CHECK(!layout.has_value(), "`layout` is not supported now.");
  if (pin_memory.value_or(false)) {
    phi::Place base_place =
        device.has_value() ? device.value()._PD_GetInner() : phi::CPUPlace();
    phi::Place pinned_place = phi::is_xpu_place(base_place)
                                  ? phi::Place(phi::XPUPinnedPlace())
                                  : phi::Place(phi::GPUPinnedPlace());
    auto dense =
        paddle::experimental::eye(n,
                                  /*num_columns=*/-1,
                                  compat::_PD_AtenScalarTypeToPhiDataType(
                                      dtype.value_or(c10::get_default_dtype())),
                                  phi::CPUPlace());
    return dense.copy_to(pinned_place, /*blocking=*/true);
  }
  return paddle::experimental::eye(
      n,
      /*num_columns=*/-1,
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      device.value_or(at::kCPU)._PD_GetInner());
}

// eye(n, m, dtype, layout, device, pin_memory)
inline at::Tensor eye(int64_t n,
                      int64_t m,
                      ::std::optional<at::ScalarType> dtype,
                      ::std::optional<at::Layout> layout,
                      ::std::optional<at::Device> device,
                      ::std::optional<bool> pin_memory) {
  PD_CHECK(!layout.has_value(), "`layout` is not supported now.");
  if (pin_memory.value_or(false)) {
    phi::Place base_place =
        device.has_value() ? device.value()._PD_GetInner() : phi::CPUPlace();
    phi::Place pinned_place = phi::is_xpu_place(base_place)
                                  ? phi::Place(phi::XPUPinnedPlace())
                                  : phi::Place(phi::GPUPinnedPlace());
    auto dense =
        paddle::experimental::eye(n,
                                  m,
                                  compat::_PD_AtenScalarTypeToPhiDataType(
                                      dtype.value_or(c10::get_default_dtype())),
                                  phi::CPUPlace());
    return dense.copy_to(pinned_place, /*blocking=*/true);
  }
  return paddle::experimental::eye(
      n,
      m,
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      device.value_or(at::kCPU)._PD_GetInner());
}

}  // namespace at
