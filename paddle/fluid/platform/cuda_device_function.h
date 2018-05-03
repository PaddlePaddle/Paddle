/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <cuda.h>

namespace paddle {
namespace platform {

// __shfl_down and __shfl have been deprecated as of CUDA 9.0.
#if CUDA_VERSION < 9000
template <typename T>
__forceinline__ __device__ T __shfl_down_sync(unsigned, T val, int delta) {
  return __shfl_down(val, delta);
}

template <typename T>
__forceinline__ __device__ T __shfl_sync(unsigned, T val, int src_line,
                                         int width) {
  return __shfl(val, src_line, width);
}
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
template <typename T>
__forceinline__ __device__ T __shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

template <typename T>
__forceinline__ __device__ T __shfl_sync(unsigned mask, T val, int src_line,
                                         int width) {
  return __shfl_sync(mask, val, src_line, width);
}
#endif

template <typename T>
__device__ T reduceSum(T val, int tid, int len) {
  // NOTE(zcd): The warp size should be taken from the
  // parameters of the GPU but not specified as 32 simply.
  // To make the reduceSum more efficiently,
  // I use Warp-Level Parallelism and assume the Warp size
  // is 32 which may be different for different GPU,
  // but most card's warp size is 32.
  const int warpSize = 32;
  __shared__ T shm[warpSize];
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, tid < len);

  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += platform::__shfl_down_sync(mask, val, offset);

  if (tid < warpSize) shm[tid] = 0;

  if (tid % warpSize == 0) {
    shm[tid / warpSize] = val;
  }
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += platform::__shfl_down_sync(mask, val, offset);
  }
  return val;
}

}  // namespace platform
}  // namespace paddle
