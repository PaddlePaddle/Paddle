// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

static constexpr int kNumCUDAThreads = 512;
static constexpr int64_t kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int64_t N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__global__ void ShuffleChannel(const int64_t nthreads,
                               const int64_t feature_map_size,
                               T* output,
                               const T* input,
                               int64_t group_row,
                               int64_t group_column,
                               int64_t len) {
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  int64_t offset =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  for (int64_t ii = index; ii < nthreads; ii += offset) {
    const int64_t n = ii / group_row / group_column / len;
    const int64_t i = (ii / group_column / len) % group_row;
    const int64_t j = ii / len % group_column;
    const int64_t k =
        ii - (n * feature_map_size + (i * group_column + j) * len);
    T* p_o = output + n * feature_map_size + (j * group_row + i) * len;
    p_o[k] = input[ii];
  }
}

}  // namespace phi
