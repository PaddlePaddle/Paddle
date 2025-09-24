// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// CUDA, XPU and HIP use same api
#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#include "paddle/phi/kernels/primitive/kernel_primitives.h"
namespace kps = phi::kps;

namespace phi {
namespace funcs {

constexpr int kMaxRank = phi::DDim::kMaxRank;

namespace details {
// Convert dims from vector to array
template <typename T, size_t ElementCount, typename VectorLikeType>
static inline Array<T, ElementCount> VectorToArray(const VectorLikeType& vec) {
  PADDLE_ENFORCE_LE(vec.size(),
                    ElementCount,
                    common::errors::InvalidArgument(
                        "Vector to Array: size not match. Received "
                        "vec.size() %d > ElementCount %d.",
                        vec.size(),
                        ElementCount));
  size_t n = static_cast<size_t>(vec.size());
  Array<T, ElementCount> ret;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = vec[i];
  }
  return ret;
}
}  // namespace details

template <typename IndexType>
struct IndexCalculator {
  IndexCalculator(int dim,
                  const std::vector<int64_t>& cal_dims,
                  const std::vector<int64_t>& cal_strides,
                  const std::vector<int64_t>& full_strides)
      : dim(dim) {
    std::vector<int64_t> dim_strides;
    for (auto i : cal_dims) {
      dim_strides.push_back(full_strides[i]);
    }
    strides = details::VectorToArray<int64_t, kMaxRank>(dim_strides);
#ifdef PADDLE_WITH_XPU_KP
    reduce_strides = details::VectorToArray<int64_t, kMaxRank>(cal_strides);
#else
    std::vector<FastDivMod<IndexType>> cal_divmoders;
    for (auto i : cal_strides) {
      cal_divmoders.emplace_back(i);
    }
    divmoders =
        details::VectorToArray<FastDivMod<IndexType>, kMaxRank>(cal_divmoders);
#endif
  }

  __device__ inline IndexType operator()(IndexType offset) const {
    IndexType index = 0;
#pragma unroll
    for (int i = 0; i < kMaxRank; ++i) {
      if (i == dim) {
        break;
      }
#ifdef PADDLE_WITH_XPU_KP
      index += (offset / reduce_strides[i]) * strides[i];
      offset = offset % reduce_strides[i];
#else
      auto divmod = divmoders[i].Divmod(offset);
      index += (divmod.val[0] * strides[i]);
      offset = divmod.val[1];
#endif
    }
    return index;
  }

  int dim;
  Array<int64_t, kMaxRank> strides;
#ifdef PADDLE_WITH_XPU_KP
  Array<int64_t, kMaxRank> reduce_strides;
#else
  Array<FastDivMod<IndexType>, kMaxRank> divmoders;
#endif
};

#endif
}  // namespace funcs
}  // namespace phi
