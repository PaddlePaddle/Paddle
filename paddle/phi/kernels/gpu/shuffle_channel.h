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
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__global__ void ShuffleChannel(const int nthreads,
                               const int feature_map_size,
                               T* output,
                               const T* input,
                               int group_row,
                               int group_column,
                               int len) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (size_t ii = index; ii < nthreads; ii += offset) {
    const int n = index / group_row / group_column / len;
    const int i = (index / group_column / len) % group_row;
    const int j = index / len % group_column;
    const int k = index - (n * feature_map_size + (i * group_column + j) * len);
    T* p_o = output + n * feature_map_size + (j * group_row + i) * len;
    p_o[k] = input[index];
  }
}

}  // namespace phi
