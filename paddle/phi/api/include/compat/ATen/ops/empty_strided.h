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
#include <c10/util/ArrayRef.h>
#include <utils/pinned_place.h>

#include <limits>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

namespace at {

namespace detail {

// PyTorch's `empty_strided_cpu` forwards `options.pinned_memory_opt()` into
// `GetCPUAllocatorMaybePinned`, so `pin_memory=true` yields page-locked
// memory. Paddle's `empty` API has no allocator hook, so follow the
// `empty_like.h` convention instead: allocate on CPU first and move the
// storage to the pinned place.
inline paddle::Tensor _PD_EmptyStridedFlatTensor(
    int64_t storage_elems,
    phi::DataType dtype,
    const at::TensorOptions& options) {
  const paddle::experimental::IntArray shape(
      std::vector<int64_t>{storage_elems});
  if (!options.pinned_memory()) {
    return paddle::experimental::empty(shape, dtype, options._PD_GetPlace());
  }
  // Pinning memory is only supported for CPU tensors.
  TORCH_CHECK(
      !options.has_device() || options.device().is_cpu(),
      "pin_memory=true requires device to be CPU, but got non-CPU device");
  auto flat_cpu = paddle::experimental::empty(shape, dtype, phi::CPUPlace());
  const phi::Place pinned_place =
      compat::_PD_GetCreatePinnedPlace(options._PD_GetPlace());
  return flat_cpu.copy_to(pinned_place, /*blocking=*/true);
}

}  // namespace detail

inline at::Tensor empty_strided(at::IntArrayRef size,
                                at::IntArrayRef stride,
                                at::TensorOptions options = {}) {
  // Match PyTorch's semantics (`computeStorageNbytes` in
  // aten/src/ATen/EmptyTensor.cpp): the storage must cover the largest
  // offset reachable through the strides, i.e.
  // `1 + sum((size[i] - 1) * stride[i])` elements (0 if any dim is 0).
  // This can exceed `numel()` for padded layouts (e.g. TMA-aligned tensors),
  // so allocating by `size` alone would under-allocate and cause OOB writes.
  TORCH_CHECK(size.size() == stride.size(),
              "dimensionality of sizes (",
              size.size(),
              ") must match dimensionality of strides (",
              stride.size(),
              ")");

  // PyTorch's `empty_strided` only supports the strided layout.
  TORCH_CHECK(options.layout() == at::kStrided,
              "empty_strided only supports strided layout, got: ",
              options.layout());

  // Match PyTorch's `_empty_strided_generic`, which runs
  // `check_size_nonnegative` over every dimension before the storage
  // computation. The 0-size fast path below breaks out early, so folding
  // this check into that loop would let inputs like size={0, -1} slip
  // through unvalidated.
  for (size_t i = 0; i < size.size(); ++i) {
    TORCH_CHECK(size[i] >= 0,
                "Trying to create tensor with negative dimension ",
                size[i],
                ": ",
                size);
  }

  int64_t storage_elems = 1;
  for (size_t i = 0; i < size.size(); ++i) {
    if (size[i] == 0) {
      storage_elems = 0;
      break;
    }
    TORCH_CHECK(stride[i] >= 0,
                "empty_strided: Negative strides are not supported, got ",
                stride[i],
                " at dimension ",
                i);
    const int64_t extent = size[i] - 1;
    TORCH_CHECK(extent == 0 ||
                    stride[i] <= std::numeric_limits<int64_t>::max() / extent,
                "Storage size calculation overflowed with sizes=",
                size,
                " and strides=",
                stride);
    const int64_t strided_elems = extent * stride[i];
    TORCH_CHECK(
        storage_elems <= std::numeric_limits<int64_t>::max() - strided_elems,
        "Storage size calculation overflowed with sizes=",
        size,
        " and strides=",
        stride);
    storage_elems += strided_elems;
  }

  // The allocator sizes the buffer as `storage_elems * itemsize` bytes; make
  // sure that product cannot wrap around `size_t` either.
  const auto dtype = compat::_PD_AtenScalarTypeToPhiDataType(options.dtype());
  const size_t itemsize = phi::SizeOf(dtype);
  TORCH_CHECK(
      itemsize == 0 || static_cast<size_t>(storage_elems) <=
                           std::numeric_limits<size_t>::max() / itemsize,
      "Storage size calculation overflowed with sizes=",
      size,
      " and strides=",
      stride);

  auto flat_tensor =
      detail::_PD_EmptyStridedFlatTensor(storage_elems, dtype, options);

  return paddle::experimental::as_strided(
      flat_tensor,
      std::vector<int64_t>(size.begin(), size.end()),
      std::vector<int64_t>(stride.begin(), stride.end()));
}

}  // namespace at
