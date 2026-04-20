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
#include <utils/dense_sparse_conversion.h>
#include <utils/pinned_place.h>

#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/common/place.h"

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

  paddle::Tensor dense;
  if (options.pinned_memory()) {
    // Pinning memory is only supported for CPU tensors
    if (options.has_device() && !options.device().is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    auto dense_cpu = paddle::experimental::zeros_like(
        base, compat::_PD_AtenScalarTypeToPhiDataType(dtype), phi::CPUPlace());
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(base_place);
    dense = dense_cpu.copy_to(pinned_place, /*blocking=*/true);
  } else {
    dense = paddle::experimental::zeros_like(
        base,
        compat::_PD_AtenScalarTypeToPhiDataType(dtype),
        options._PD_GetPlace());
  }
  return compat::_PD_ConvertToSparseIfNeeded(dense, layout);
}

inline at::Tensor zeros_like(const at::Tensor& self,
                             ::std::optional<at::ScalarType> dtype,
                             ::std::optional<at::Layout> layout,
                             ::std::optional<at::Device> device,
                             ::std::optional<bool> pin_memory,
                             ::std::optional<at::MemoryFormat> memory_format) {
  PD_CHECK(!(memory_format.has_value() &&
             memory_format.value() != c10::MemoryFormat::Contiguous),
           "`MemoryFormat` other than Contiguous is not supported now.");

  auto options = at::TensorOptions()
                     .dtype(dtype)
                     .layout(layout)
                     .device(device)
                     .pinned_memory(pin_memory);
  return zeros_like(self, options, memory_format);
}

}  // namespace at
