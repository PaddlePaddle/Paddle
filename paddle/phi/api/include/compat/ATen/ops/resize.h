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
#include "paddle/phi/core/memory/malloc.h"

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

inline size_t ResizeCheckedStorageBytes(int64_t numel,
                                        size_t itemsize,
                                        size_t storage_offset_bytes) {
  const auto numel_size = static_cast<size_t>(numel);
  TORCH_CHECK(
      itemsize == 0 || numel_size <= (std::numeric_limits<size_t>::max() -
                                      storage_offset_bytes) /
                                         itemsize,
      "resize_ size is too large in bytes");
  return storage_offset_bytes + numel_size * itemsize;
}

}  // namespace detail

// resize_ - operate on the underlying DenseTensor directly so we preserve
// storage semantics across shrink/grow round-trips. When growth exceeds the
// current capacity, expand the shared storage itself so aliasing views keep
// their storage offset and existing storage contents stay intact.
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
  const size_t new_storage_bytes = detail::ResizeCheckedStorageBytes(
      new_numel, itemsize, dense_tensor->meta().offset);
  const size_t current_storage_bytes =
      dense_tensor->Holder() == nullptr ? 0 : dense_tensor->Holder()->size();

  if (new_storage_bytes <= current_storage_bytes || new_numel == 0) {
    dense_tensor->Resize(dims);
    return *this;
  }

  // Sync through the compat Storage path first so the DenseTensor holder is a
  // live StorageHolderView backed by shared StorageImpl.
  auto storage = this->storage();
  const auto old_holder = dense_tensor->Holder();
  TORCH_CHECK(old_holder != nullptr,
              "resize_ cannot grow a tensor without allocated storage");
  const phi::Place place = old_holder->place();
  auto new_holder = paddle::memory::AllocShared(place, new_storage_bytes);
  TORCH_CHECK(new_holder != nullptr, "resize_ failed to allocate storage");
  const size_t copy_bytes = std::min(old_holder->size(), new_storage_bytes);
  if (copy_bytes > 0 && old_holder->ptr() != nullptr &&
      old_holder->ptr() != new_holder->ptr()) {
    phi::memory_utils::Copy(
        place, new_holder->ptr(), place, old_holder->ptr(), copy_bytes);
  }
  storage.set_data_ptr_noswap(std::move(new_holder));
  dense_tensor->Resize(phi::make_ddim(dims));
  return *this;
}

}  // namespace at
