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

// A 0-Size index selects nothing, so the indexed region is empty no matter
// what the other operands look like.
inline bool HasEmptyIndex(const std::vector<const DenseTensor*>& index) {
  return std::any_of(index.begin(), index.end(), [](const DenseTensor* t) {
    return t->numel() == 0;
  });
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
