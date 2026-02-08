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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>

#ifdef __HIPCC__
#define __syncwarp() __all(1)
#endif

namespace phi {

#ifdef __HIPCC__
#define THREADS_PER_BLOCK 64
#else
#define THREADS_PER_BLOCK 32
#endif
#define FULL_MASK 0xffffffff

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
  for (int offset = 16; offset > 0; offset /= 2) {
#ifdef __HIPCC__
    val += __shfl_down(val, offset);
#else
    val += __shfl_down_sync(FULL_MASK, val, offset);
#endif
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
#ifdef __HIPCC__
  static __shared__ T shared[64];
#else
  static __shared__ T shared[32];
#endif
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);
  __syncthreads();
  if (lane == 0) shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid == 0) val = warpReduceSum(val);

  return val;
}

template <typename T>
__global__ void set_zero(T *x, int64_t num) {
  for (int64_t i =
           static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
       i < num;
       i += blockDim.x * gridDim.x)
    x[i] = static_cast<T>(0);
}

template <typename T>
__global__ void channel_first(const T *input,
                              T *rinput,
                              const int64_t N,
                              const int64_t channel,
                              const int64_t H,
                              const int64_t W,
                              const int pad_size) {
  int64_t global_idx = static_cast<int64_t>(blockIdx.x);
  int64_t stride = static_cast<int64_t>(gridDim.x);

  int p_H = H + 2 * pad_size;
  int p_W = W + 2 * pad_size;
  int64_t p_dimcw = channel * p_W;
  int64_t p_dimchw = channel * p_H * p_W;

  while (global_idx < int64_t(N) * H * W) {
    int64_t idx = global_idx;
    int64_t n = idx / (H * W);
    idx = idx % (H * W);
    int64_t h = idx / W;
    int64_t w = idx % W;

    for (int64_t c = threadIdx.x; c < channel; c += blockDim.x) {
      rinput[n * p_dimchw + (h + pad_size) * p_dimcw +
             (w + pad_size) * channel + c] =
          input[n * (channel * H * W) + c * (H * W) + h * W + w];
    }

    global_idx += stride;
  }
}

}  // namespace phi
