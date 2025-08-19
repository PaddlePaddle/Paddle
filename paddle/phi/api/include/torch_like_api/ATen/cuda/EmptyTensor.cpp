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

#include <ATen/cuda/EmptyTensor.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBody.h>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

namespace at::detail {

at::Tensor empty_cuda(IntArrayRef size,
                      ScalarType dtype,
                      std::optional<Device> device_opt,
                      std::optional<c10::MemoryFormat> memory_format_opt) {
  if (memory_format_opt.has_value()) {
    // Restriding a just-created empty contiguous tensor does nothing.
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      // LOG(ERROR) << "empty_cuda not support memory_format_opt
      // !=MemoryFormat::Contiguous now";
    }
  }
  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(dtype),
      phi::GPUPlace());
}

// TensorBase empty_cuda(
//     IntArrayRef size,
//     std::optional<ScalarType> dtype_opt,
//     std::optional<Layout> layout_opt,
//     std::optional<Device> device_opt,
//     std::optional<bool> pin_memory_opt,
//     std::optional<c10::MemoryFormat> memory_format_opt) {
//   TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense
//   CPU tensors can be pinned");
//   TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) ==
//   Layout::Strided);

//   const auto dtype = dtype_or_default(dtype_opt);
//   return at::detail::empty_cuda(size, dtype, device_opt, memory_format_opt);
// }

at::Tensor empty_cuda(IntArrayRef size, const TensorOptions &options) {
  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype_opt().value()),
      phi::GPUPlace());
}

}  // namespace at::detail
