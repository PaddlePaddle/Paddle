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
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/ddim.h"

namespace at {

namespace detail {

inline int64_t ResizeCheckedNumel(at::IntArrayRef size) {
  int64_t numel = 1;
  for (const auto dim : size) {
    TORCH_CHECK(dim >= 0,
                "Trying to create tensor with negative dimension ",
                dim,
                ": ",
                size);
    if (dim == 0) {
      numel = 0;
      continue;
    }
    TORCH_CHECK(numel <= std::numeric_limits<int64_t>::max() / dim,
                "resize_ size is too large, possible overflow for size ",
                size);
    numel *= dim;
  }
  return numel;
}

}  // namespace detail

// resize_ - operate on the underlying DenseTensor directly so we preserve
// storage semantics across shrink/grow round-trips and only reallocate when
// the requested shape exceeds the current storage capacity.
inline const at::Tensor& Tensor::resize_(
    at::IntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) const {
  // Keep old compat behavior for memory_format in this split PR.
  // TODO(youge325): add real ChannelsLast/ChannelsLast3d restride support
  // later.
  (void)memory_format;

  std::vector<int64_t> dims(size.begin(), size.end());
  int64_t new_numel = detail::ResizeCheckedNumel(size);
  auto dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensor_.impl());
  TORCH_CHECK(dense_tensor != nullptr,
              "resize_ only supports DenseTensor, but got a non-dense tensor");
  TORCH_CHECK(tensor_.defined(),
              "resize_ is not allowed on an undefined tensor");

  const size_t itemsize = phi::SizeOf(dense_tensor->dtype());
  const size_t old_numel = static_cast<size_t>(tensor_.numel());
  const size_t new_numel_size = static_cast<size_t>(new_numel);
  const size_t required_bytes = new_numel_size * itemsize;
  const size_t available_bytes =
      dense_tensor->Holder() == nullptr
          ? 0
          : dense_tensor->Holder()->size() - dense_tensor->meta().offset;

  if (required_bytes <= available_bytes || new_numel == 0) {
    dense_tensor->Resize(dims);
    return *this;
  }

  const auto old_holder = dense_tensor->Holder();
  TORCH_CHECK(old_holder != nullptr,
              "resize_ cannot grow a tensor without allocated storage");
  const size_t old_offset = dense_tensor->meta().offset;
  const size_t copy_bytes = std::min(old_numel, new_numel_size) * itemsize;
  const phi::Place place = old_holder->place();
  const void* old_data =
      old_holder == nullptr
          ? nullptr
          : reinterpret_cast<const uint8_t*>(old_holder->ptr()) + old_offset;

  dense_tensor->ResizeAndAllocate(phi::make_ddim(dims));
  void* new_data = dense_tensor->data();
  if (copy_bytes > 0 && old_data != nullptr && old_data != new_data) {
    phi::memory_utils::Copy(place, new_data, place, old_data, copy_bytes);
  }
  return *this;
}

}  // namespace at
