/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {
namespace funcs {

constexpr int MAX_DIMS = DDim::kMaxRank;

// The GPU offset calculator instantiated with signed_strides=true keeps its
// offsets in std::make_signed_t<INDEX_T>, so its 32-bit fast path is bounded by
// int32_t instead of uint32_t. Byte extents in (2 GiB, 4 GiB] must therefore
// fall through to the 64-bit path instead of reusing IsInUint32Range.
constexpr bool IsInInt32Range(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

template <typename... Values>
constexpr bool IsInInt32Range(int64_t value, Values... values) {
  return IsInInt32Range(value) && (IsInInt32Range(values) && ...);
}

// Lowest and highest offset, relative to an operand's base, that walking
// `dims` with `strides` can produce. Both ends have to be tracked separately:
// a reversed axis carries a negative stride and pulls `lo` below the base,
// and once an operand is strided `numel` bounds neither end.
struct OperandReach {
  int64_t lo = 0;
  int64_t hi = 0;
};

// Accumulates one operand's reach into `reach`, so that operands sharing a base
// (x and the index tensor are both offset from slice_offset) can be combined.
// `scale` converts `strides` into the unit the caller wants, e.g. sizeof(dtype)
// to turn element strides into bytes.
inline void AccumulateReach(int64_t ndim,
                            const int64_t* dims,
                            const int64_t* strides,
                            int64_t scale,
                            OperandReach* reach) {
  for (int64_t i = 0; i < ndim; ++i) {
    if (dims[i] <= 1) {
      continue;  // a 1-size axis never uses its stride, a 0-size one is empty
    }
    const int64_t r = (dims[i] - 1) * strides[i] * scale;
    if (r < 0) {
      reach->lo += r;
    } else {
      reach->hi += r;
    }
  }
}

// Largest |byte offset| an offset calculator can produce for an operand it
// walks with the tensor's own dims and strides, i.e. hi - lo of its reach.
// `numel * sizeof(dtype)` only bounds this for a contiguous tensor. The put
// kernels hand the `value` operand to the calculator with its own strides, and
// its element size is unrelated to the ones already covered by the dispatch:
// a complex128 value of 2e8 elements reaches 3.2e9 bytes while x, out and the
// int64 index operand (1.6e9 bytes) all stay inside int32_t, so the 32-bit
// offset type would wrap on a shape the 64-bit path handles.
inline int64_t StridedOperandByteSpan(const DenseTensor& t) {
  OperandReach reach;
  AccumulateReach(t.dims().size(),
                  t.dims().Get(),
                  t.strides().Get(),
                  static_cast<int64_t>(SizeOf(t.dtype())),
                  &reach);
  return reach.hi - reach.lo;
}

// Byte extent of the index operand, i.e. the third operand of the offset
// calculators built by the index_elementwise kernels. Its strides come from
// the broadcast index shape (the tail of `index_dims`, see cal_shape_stride)
// scaled by sizeof(int64_t), so it reaches (elements - 1) * sizeof(int64_t)
// bytes. That can exceed int32_t while x and out stay well inside it -- e.g.
// a bool x[1] gathered by an int64 index of 3e8 elements -- so it has to take
// part in the 32/64 bit dispatch, otherwise the signed calculator's
// CheckOffsetRange rejects a shape the unsigned path used to handle.
inline int64_t IndexOperandByteSpan(const std::vector<int64_t>& index_dims) {
  auto shape_begin = std::find(index_dims.begin(), index_dims.end(), -1);
  if (shape_begin == index_dims.end()) {
    return 0;
  }
  int64_t elements = 1;
  for (++shape_begin; shape_begin != index_dims.end() && *shape_begin != -1;
       ++shape_begin) {
    elements *= *shape_begin;
  }
  if (elements == 0) {
    return 0;
  }
  return (elements - 1) * static_cast<int64_t>(sizeof(int64_t));
}

// A 0-Size index selects nothing, so the indexed region is empty no matter
// what the other operands look like.
inline bool HasEmptyIndex(const std::vector<const DenseTensor*>& index) {
  return std::any_of(index.begin(), index.end(), [](const DenseTensor* t) {
    return t->numel() == 0;
  });
}

// Rewrites the reversed axes of a strided operand into forward ones.
//
// A reversed view (`x[::-1]`) carries a negative stride and its base points at
// the *highest* address of that axis. Backends whose gather/scatter primitives
// can only walk an operand forwards cannot consume that description, so on each
// axis `m` with `S[m] < 0` we substitute `i_m -> dims[m]-1-i_m`. That moves the
// base to the low end of the axis (`byte_offset += (dims[m]-1)*S[m]*elem_size`,
// which stays >= 0 because it only walks back to the lowest address the view
// already reaches) and turns `S[m]` into `|S[m]|`.
//
// The substitution reverses the traversal order of `m`, so the caller has to
// flip the *dense* operand (the contiguous output or value buffer) along the
// axes returned in `flip_axes` to compensate.
inline void NormalizeNegativeStrides(const std::vector<int64_t>& dims,
                                     int64_t elem_size,
                                     std::vector<int64_t>* strides,
                                     int64_t* byte_offset,
                                     std::vector<int64_t>* flip_axes) {
  for (size_t i = 0; i < strides->size(); ++i) {
    if ((*strides)[i] >= 0) {
      continue;
    }
    // A 1-size axis never uses its stride, so reversing it is a no-op and the
    // dense operand must not be flipped along it.
    if (dims[i] > 1) {
      *byte_offset += (dims[i] - 1) * (*strides)[i] * elem_size;
      flip_axes->push_back(static_cast<int64_t>(i));
    }
    (*strides)[i] = -(*strides)[i];
  }
}

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename IndexT>
std::array<char*, DDim::kMaxRank> GetIndexDataPtrs(
    const std::vector<const DenseTensor*>& index) {
  std::array<char*, DDim::kMaxRank> index_ptrs{};

  PADDLE_ENFORCE_LE(index.size(),
                    DDim::kMaxRank,
                    "The number of index tensors exceeds the maximum rank.");

  for (size_t i = 0; i < index.size(); ++i) {
    // A 0-Size index tensor legally has no data pointer. The iteration space
    // of every caller is broadcast against the index shape, so it is empty as
    // well and the pointer is never dereferenced.
    if (index[i]->numel() == 0) {
      index_ptrs[i] = nullptr;
      continue;
    }

    const IndexT* p_index = index[i]->data<IndexT>();

    PADDLE_ENFORCE_NOT_NULL(
        p_index,
        ::common::errors::InvalidArgument(
            "The pointer p_index is nullptr, "
            "please check whether the index tensor is valid and "
            "its data is correctly initialized."));

    index_ptrs[i] = reinterpret_cast<char*>(const_cast<IndexT*>(p_index));
  }

  return index_ptrs;
}

}  // namespace funcs
}  // namespace phi
