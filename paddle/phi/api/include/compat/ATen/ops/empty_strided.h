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
#include <limits>
#include <optional>
#include <string_view>

#include "paddle/phi/api/include/api.h"

namespace at {

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

  int64_t storage_elems = 1;
  for (size_t i = 0; i < size.size(); ++i) {
    TORCH_CHECK(size[i] >= 0,
                "Trying to create tensor with negative dimension ",
                size[i],
                ": ",
                size);
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

  auto flat_tensor = paddle::experimental::empty(
      paddle::experimental::IntArray(std::vector<int64_t>{storage_elems}),
      compat::_PD_AtenScalarTypeToPhiDataType(options.dtype()),
      options._PD_GetPlace());

  return paddle::experimental::as_strided(
      flat_tensor,
      std::vector<int64_t>(size.begin(), size.end()),
      std::vector<int64_t>(stride.begin(), stride.end()));
}

}  // namespace at
