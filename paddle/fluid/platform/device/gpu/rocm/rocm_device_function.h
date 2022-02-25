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

// NOTE(): support float16 to half in header file.
#define PADDLE_CUDA_FP16
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace platform {

#define CREATE_SHFL_MASK(mask, predicate) mask = __ballot((predicate))

#define CUDA_LAUNCH_KERNEL_BASE(dim, ...)  \
  case (dim): {                            \
    constexpr auto kPowerOfTwoDim = (dim); \
    __VA_ARGS__;                           \
  } break

#define CUDA_LAUNCH_KERNEL_HELPER(...)          \
  CUDA_LAUNCH_KERNEL_BASE(1024, ##__VA_ARGS__); \
  CUDA_LAUNCH_KERNEL_BASE(512, ##__VA_ARGS__);  \
  CUDA_LAUNCH_KERNEL_BASE(256, ##__VA_ARGS__);  \
  CUDA_LAUNCH_KERNEL_BASE(128, ##__VA_ARGS__);  \
  CUDA_LAUNCH_KERNEL_BASE(64, ##__VA_ARGS__);   \
  CUDA_LAUNCH_KERNEL_BASE(32, ##__VA_ARGS__);

template <typename T>
__forceinline__ __device__ T CudaShuffleDownSync(unsigned mask, T val,
                                                 int delta,
                                                 int width = warpSize) {
  return __shfl_down(val, delta, width);
}

template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask, T val,
                                                int width = warpSize) {
  return __shfl_xor(val, width);
}

template <>
__forceinline__ __device__ float16 CudaShuffleDownSync(unsigned mask,
                                                       float16 val, int delta,
                                                       int width) {
  return float16(__shfl_down(static_cast<float>(val),
                             static_cast<unsigned>(delta), width));
}

template <>
__forceinline__ __device__ bfloat16 CudaShuffleDownSync(unsigned mask,
                                                        bfloat16 val, int delta,
                                                        int width) {
  return bfloat16(__shfl_down(static_cast<float>(val),
                              static_cast<unsigned>(delta), width));
}

template <>
__forceinline__ __device__ paddle::platform::complex<float> CudaShuffleDownSync(
    unsigned mask, paddle::platform::complex<float> val, int delta, int width) {
  float real = __shfl_down(val.real, delta, width);
  float imag = __shfl_down(val.imag, delta, width);
  return paddle::platform::complex<float>(real, imag);
}

template <>
__forceinline__ __device__ paddle::platform::complex<double>
CudaShuffleDownSync(unsigned mask, paddle::platform::complex<double> val,
                    int delta, int width) {
  double real = __shfl_down(val.real, delta, width);
  double imag = __shfl_down(val.imag, delta, width);
  return paddle::platform::complex<double>(real, imag);
}

template <>
__forceinline__ __device__ float16 CudaShuffleXorSync(unsigned mask,
                                                      float16 val, int width) {
  return float16(__shfl_xor(static_cast<float>(val), width));
}

template <>
__forceinline__ __device__ bfloat16 CudaShuffleXorSync(unsigned mask,
                                                       bfloat16 val,
                                                       int width) {
  return bfloat16(__shfl_xor(static_cast<float>(val), width));
}

template <>
__forceinline__ __device__ paddle::platform::complex<float> CudaShuffleXorSync(
    unsigned mask, paddle::platform::complex<float> val, int width) {
  float real = __shfl_xor(val.real, width);
  float imag = __shfl_xor(val.imag, width);
  return paddle::platform::complex<float>(real, imag);
}

template <>
__forceinline__ __device__ paddle::platform::complex<double> CudaShuffleXorSync(
    unsigned mask, paddle::platform::complex<double> val, int width) {
  double real = __shfl_xor(val.real, width);
  double imag = __shfl_xor(val.imag, width);
  return paddle::platform::complex<double>(real, imag);
}

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
  return __shfl(val, src_line, width);
}

template <typename T>
HOSTDEVICE T Infinity() {
  return INFINITY;
}

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
    val += platform::CudaShuffleDownSync(mask, val, offset);

  if (tid < warpSize) shm[tid] = 0;
  __syncthreads();

  if (tid % warpSize == 0) {
    shm[tid / warpSize] = val;
  }
  __syncthreads();

  CREATE_SHFL_MASK(mask, tid < warpSize);

  if (tid < warpSize) {
    val = shm[tid];
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += platform::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

}  // namespace platform
}  // namespace paddle
