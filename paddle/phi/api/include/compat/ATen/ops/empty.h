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

#include <c10/core/TensorOptions.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

inline at::Tensor empty(
    at::IntArrayRef size,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  PD_CHECK(!(memory_format.has_value() &&
             memory_format.value() != c10::MemoryFormat::Contiguous),
           "`MemoryFormat` other than Contiguous is not supported now.");
  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());
}

inline at::Tensor empty(at::IntArrayRef size,
                        ::std::optional<at::ScalarType> dtype,
                        ::std::optional<at::Layout> layout,
                        ::std::optional<at::Device> device,
                        ::std::optional<bool> pin_memory,
                        ::std::optional<at::MemoryFormat> memory_format) {
  PD_CHECK(!layout.has_value(), "`layout` is not supported now.");
  PD_CHECK(!(pin_memory.has_value() && pin_memory.value() != false),
           "`pin_memory` other than False is not supported now.");
  PD_CHECK(!(memory_format.has_value() &&
             memory_format.value() != c10::MemoryFormat::Contiguous),
           "`MemoryFormat` other than Contiguous is not supported now.");

  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(
          dtype.value_or(c10::get_default_dtype())),
      device.value_or(at::kCPU)._PD_GetInner());
}

#define empty_symint empty  // SymIntArrayRef is same as IntArrayRef

}  // namespace at

namespace torch {
using at::empty;
}  // namespace torch
