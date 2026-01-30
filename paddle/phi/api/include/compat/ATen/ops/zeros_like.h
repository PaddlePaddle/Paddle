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

#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <c10/core/TensorOptions.h>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/sparse_api.h"

namespace at {

inline at::Tensor zeros_like(
    const at::Tensor& self,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  PD_CHECK(!(memory_format.has_value() &&
             memory_format.value() != c10::MemoryFormat::Contiguous),
           "`MemoryFormat` other than Contiguous is not supported now.");

  auto layout = options.layout();
  if (layout == c10::kStrided && (self.is_sparse() || self.is_sparse_csr())) {
    layout = self.layout();
  }

  auto dtype = options.dtype();
  if (dtype == c10::ScalarType::Undefined) {
    dtype = self.scalar_type();
  }

  paddle::Tensor base = self._PD_GetInner();
  if (self.is_sparse() || self.is_sparse_csr()) {
    base = paddle::experimental::sparse::to_dense(base);
  }

  auto dense = paddle::experimental::zeros_like(
      base,
      compat::_PD_AtenScalarTypeToPhiDataType(dtype),
      options._PD_GetPlace());
  return detail::_PD_ConvertToSparseIfNeeded(dense, layout);
}

inline at::Tensor zeros_like(const at::Tensor& self,
                             ::std::optional<at::ScalarType> dtype,
                             ::std::optional<at::Layout> layout,
                             ::std::optional<at::Device> device,
                             ::std::optional<bool> pin_memory,
                             ::std::optional<at::MemoryFormat> memory_format) {
  PD_CHECK(!(pin_memory.has_value() && pin_memory.value() != false),
           "`pin_memory` other than False is not supported now.");
  PD_CHECK(!(memory_format.has_value() &&
             memory_format.value() != c10::MemoryFormat::Contiguous),
           "`MemoryFormat` other than Contiguous is not supported now.");

  auto resolved_layout = layout.value_or(
      (self.is_sparse() || self.is_sparse_csr()) ? self.layout()
                                                 : c10::kStrided);
  auto resolved_dtype = dtype.value_or(self.scalar_type());
  auto resolved_device = device.value_or(self.device());

  paddle::Tensor base = self._PD_GetInner();
  if (self.is_sparse() || self.is_sparse_csr()) {
    base = paddle::experimental::sparse::to_dense(base);
  }

  auto dense = paddle::experimental::zeros_like(
      base,
      compat::_PD_AtenScalarTypeToPhiDataType(resolved_dtype),
      resolved_device._PD_GetInner());
  return detail::_PD_ConvertToSparseIfNeeded(dense, resolved_layout);
}

}  // namespace at
namespace torch {
using at::zeros_like;
}  // namespace torch
