// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>

#define CUDA_CHECK(cmd)                                                  \
  do {                                                                   \
    cudaError_t e = cmd;                                                 \
    if (e != cudaSuccess) {                                              \
      std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << " " \
                << cudaGetErrorString(e) << std::endl;                   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CURAND_CHECK(cmd)                                                  \
  do {                                                                     \
    curandStatus_t error = (cmd);                                          \
    if (error != CURAND_STATUS_SUCCESS) {                                  \
      std::cout << "CuRAND failure " << __FILE__ << ":" << __LINE__ << " " \
                << std::endl;                                              \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

#if CUDART_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
#if CUDART_VERSION < 9000
  return __shfl(val, src_line, width);
#else
  return __shfl_sync(mask, val, src_line, width);
#endif
}

template <typename T>
__forceinline__ __device__ T CudaShuffleUpSync(unsigned mask, T val, int delta,
                                               int width = 32) {
#if CUDART_VERSION < 9000
  return __shfl_up(val, delta, width);
#else
  return __shfl_up_sync(mask, val, delta, width);
#endif
}

const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

void RandomizeFloat(void* dest, const int count, const int seed);
void FeedInputFloat(float* dest, const int count, const float* src,
                    const int size);
