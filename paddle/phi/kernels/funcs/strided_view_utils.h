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

#include <cstdint>
#include <limits>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

// int64_t arithmetic that reports overflow instead of wrapping around.
//
// The element range of a strided view is computed from a caller supplied
// (shape, stride, offset) triple, so the products and sums below can exceed
// int64_t for adversarial input. Without these guards the range check wraps
// around and happily accepts a view that reaches far outside of the
// allocation, which is worse than having no check at all. __int128 and
// __builtin_*_overflow are avoided on purpose: this header is reachable from
// the public compat headers, which are also compiled by MSVC.
inline bool SafeMulInt64(int64_t a, int64_t b, int64_t* out) {
  constexpr int64_t kMax = (std::numeric_limits<int64_t>::max)();
  constexpr int64_t kMin = (std::numeric_limits<int64_t>::lowest)();
  if (a == 0 || b == 0) {
    *out = 0;
    return true;
  }
  // Handled separately so that the divisions below never evaluate kMin / -1.
  if (a == -1) {
    if (b == kMin) return false;
    *out = -b;
    return true;
  }
  if (b == -1) {
    if (a == kMin) return false;
    *out = -a;
    return true;
  }
  if (a > 0) {
    if (b > 0 ? a > kMax / b : b < kMin / a) return false;
  } else {
    if (b > 0 ? a < kMin / b : a < kMax / b) return false;
  }
  *out = a * b;
  return true;
}

inline bool SafeAddInt64(int64_t a, int64_t b, int64_t* out) {
  constexpr int64_t kMax = (std::numeric_limits<int64_t>::max)();
  constexpr int64_t kMin = (std::numeric_limits<int64_t>::lowest)();
  if (b > 0 && a > kMax - b) return false;
  if (b < 0 && a < kMin - b) return false;
  *out = a + b;
  return true;
}

// Element indices, relative to the start of the allocation, that a strided
// view touches. `empty` marks a view with a zero sized dimension, which reads
// and writes nothing at all.
struct StridedViewRange {
  int64_t min_index{0};
  int64_t max_index{0};
  bool empty{false};
};

inline StridedViewRange ComputeStridedViewRange(
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& strides,
    int64_t base_index) {
  StridedViewRange range;
  range.min_index = base_index;
  range.max_index = base_index;
  for (size_t i = 0; i < dims.size(); ++i) {
    PADDLE_ENFORCE_GE(dims[i],
                      0,
                      common::errors::InvalidArgument(
                          "The shape of a strided view must be non-negative, "
                          "but got %s.",
                          common::make_ddim(dims)));
    if (dims[i] == 0) {
      range.empty = true;
      return range;
    }
    int64_t span = 0;
    bool ok = SafeMulInt64(strides[i], dims[i] - 1, &span);
    if (ok) {
      ok = span > 0 ? SafeAddInt64(range.max_index, span, &range.max_index)
                    : SafeAddInt64(range.min_index, span, &range.min_index);
    }
    PADDLE_ENFORCE_EQ(
        ok,
        true,
        common::errors::InvalidArgument(
            "The element range of the view described by shape %s, stride %s "
            "and element offset %d overflows int64.",
            common::make_ddim(dims),
            common::make_ddim(strides),
            base_index));
  }
  return range;
}

// Rejects views that would reach outside of `input`'s allocation. Without this
// check a bad (shape, stride, offset) triple silently produces a tensor whose
// reads and writes corrupt neighbouring heap memory.
//
// `offset` is a byte offset into the allocation, matching AsStridedKernel and
// DenseTensorMeta::offset.
inline void ValidateStridedViewStorage(const std::vector<int64_t>& dims,
                                       const std::vector<int64_t>& strides,
                                       int64_t offset,
                                       const DenseTensor& input) {
  if (input.numel() == 0 || input.Holder() == nullptr) {
    return;
  }
  PADDLE_ENFORCE_EQ(dims.size(),
                    strides.size(),
                    common::errors::InvalidArgument(
                        "The size of dims and strides should be equal."));
  const int64_t itemsize = static_cast<int64_t>(SizeOf(input.dtype()));
  PADDLE_ENFORCE_EQ(offset % itemsize,
                    0,
                    common::errors::InvalidArgument(
                        "The offset(%d) is a byte offset and must be a "
                        "multiple of the element size(%d) of the input.",
                        offset,
                        itemsize));
  const StridedViewRange range =
      ComputeStridedViewRange(dims, strides, offset / itemsize);
  if (range.empty) {
    return;
  }
  const int64_t storage_numel =
      static_cast<int64_t>(input.Holder()->size()) / itemsize;
  PADDLE_ENFORCE_GE(range.min_index,
                    0,
                    common::errors::InvalidArgument(
                        "The view described by shape %s, stride %s and offset "
                        "%d reaches element %d, which is before the beginning "
                        "of the input storage.",
                        common::make_ddim(dims),
                        common::make_ddim(strides),
                        offset,
                        range.min_index));
  PADDLE_ENFORCE_LT(range.max_index,
                    storage_numel,
                    common::errors::InvalidArgument(
                        "The view described by shape %s, stride %s and offset "
                        "%d reaches element %d, but the input storage only "
                        "holds %d elements.",
                        common::make_ddim(dims),
                        common::make_ddim(strides),
                        offset,
                        range.max_index,
                        storage_numel));
}

}  // namespace phi
