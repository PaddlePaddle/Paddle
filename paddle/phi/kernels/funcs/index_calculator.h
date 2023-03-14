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
static inline phi::Array<T, ElementCount> VectorToArray(
    const VectorLikeType& vec) {
  PADDLE_ENFORCE_LE(
      vec.size(),
      ElementCount,
      phi::errors::InvalidArgument("Vector to Array: size not match. Received "
                                   "vec.size() %d > ElementCount %d.",
                                   vec.size(),
                                   ElementCount));
  size_t n = static_cast<size_t>(vec.size());
  phi::Array<T, ElementCount> ret;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = vec[i];
  }
  return ret;
}
}  // namespace details
struct IndexCalculator {
  IndexCalculator(int dim,
                  const std::vector<int>& cal_dims,
                  const std::vector<int>& cal_strides,
                  const std::vector<int>& full_strides)
      : dim(dim) {
    dims = details::VectorToArray<int, kMaxRank>(cal_dims);
    strides = details::VectorToArray<int, kMaxRank>(full_strides);
    reduce_strides = details::VectorToArray<int, kMaxRank>(cal_strides);
#ifndef PADDLE_WITH_XPU_KP
    std::vector<kps::details::FastDivMod> cal_divmoders;
    // fast divmod
    for (auto i : cal_strides) {
      cal_divmoders.push_back(kps::details::FastDivMod(i));
    }
    divmoders = details::VectorToArray<kps::details::FastDivMod, kMaxRank>(
        cal_divmoders);
#endif
  }

  __device__ inline int operator()(int offset) const {
#ifdef PADDLE_WITH_XPU_KP
    int index = 0;
#pragma unroll
    for (int i = 0; i < kMaxRank; ++i) {
      if (i == dim) {
        break;
      }
      index += (offset / reduce_strides[i]) * strides[dims[i]];
      offset = offset % reduce_strides[i];
    }
    return index;
#else
    int index = 0;
#pragma unroll
    for (int i = 0; i < kMaxRank; ++i) {
      if (i == dim) {
        break;
      }
      auto divmod = divmoders[i].Divmod(offset);
      index += (divmod.val[0] * strides[dims[i]]);
      offset = divmod.val[1];
    }
    return index;
#endif
  }

  int dim;
  phi::Array<int, kMaxRank> dims;
  phi::Array<int, kMaxRank> strides;
  phi::Array<int, kMaxRank> reduce_strides;
#ifndef PADDLE_WITH_XPU_KP
  phi::Array<kps::details::FastDivMod, kMaxRank> divmoders;
#endif
};

#endif
}  // namespace funcs
}  // namespace phi
