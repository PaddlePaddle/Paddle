/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <cuda_fp16.h>
#include <algorithm>

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__device__ __forceinline__ T FromFloat(float a);

template <typename T>
__device__ __forceinline__ float ToFloat(T a);

template <typename T>
__device__ __forceinline__ T exp_func(T a);

template <typename T>
struct KeyValuePair;

template <typename T>
using kvp = KeyValuePair<T>;

// from_float
template <>
__device__ __forceinline__ float FromFloat<float>(float a) {
  return a;
}

#ifdef SUPPORTS_CUDA_FP16
template <>
__device__ __forceinline__ half FromFloat<half>(float a) {
  return __float2half(a);
}
#endif

// to_float
template <>
__device__ __forceinline__ float ToFloat<float>(float a) {
  return a;
}

#ifdef SUPPORTS_CUDA_FP16
template <>
__device__ __forceinline__ float ToFloat<half>(half a) {
  return __half2float(a);
}
#endif

template <>
__device__ __forceinline__ float exp_func<float>(float a) {
  return expf(a);
}

#ifdef SUPPORTS_CUDA_FP16
template <>
__device__ __forceinline__ half exp_func<half>(half a) {
#if __CUDA_ARCH__ >= 600
  return hexp(a);
#else
  return FromFloat<half>(expf(ToFloat<half>(a)));
#endif
}
#endif

template <>
struct KeyValuePair<float> {
  __device__ __forceinline__ KeyValuePair() {}
  __device__ __forceinline__ KeyValuePair(float k, float v)
      : key(k), value(v) {}
  __device__ __forceinline__ KeyValuePair(const KeyValuePair &a) {
    key = a.key;
    value = a.value;
  }
  float key;
  float value;
  __device__ __forceinline__ KeyValuePair
  operator+(const KeyValuePair &a) const {
    KeyValuePair tmp;
    tmp.key = key + a.key;
    tmp.value = value + a.value;
    return tmp;
  }
};

#ifdef SUPPORTS_CUDA_FP16
template <>
struct KeyValuePair<half> {
  __device__ __forceinline__ KeyValuePair() {}
  __device__ __forceinline__ KeyValuePair(half k, half v) : key(k), value(v) {}
  __device__ __forceinline__ KeyValuePair(const KeyValuePair &a) {
    key = a.key;
    value = a.value;
  }
  half key;
  half value;
  __device__ __forceinline__ KeyValuePair
  operator+(const KeyValuePair &a) const {
    const half2 a2 = __halves2half2(key, value);
    const half2 b2 = __halves2half2(a.key, a.value);
    const half2 res = __hadd2(a2, b2);
    return KeyValuePair(res.x, res.y);
  }
};
#endif

#define FINAL_MASK 0xffffffff
#define HALF_WARP 16
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val += __shfl_xor_sync(lane_mask, val, mask, warpSize);
#else
    val += __shfl_xor(val, mask, warpSize);
#endif
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : static_cast<T>(0.0f);
  val = warpReduceSum<T>(val, mask);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val, unsigned lane_mask) {
  for (int mask = HALF_WARP; mask > 0; mask >>= 1)
#if __CUDA_ARCH__ >= 350 && CUDA_VERSION >= 9000
    val = max(val, __shfl_xor_sync(lane_mask, val, mask, warpSize));
#else
    val = max(val, __shfl_xor(val, mask, warpSize));
#endif
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val, unsigned mask) {
  static __shared__ T shared[WARP_SIZE];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax(val, mask);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // align block_span to warpSize
  int block_span = (blockDim.x + warpSize - 1) >> 5;
  val = (threadIdx.x < block_span) ? shared[lane] : -1e10f;
  val = warpReduceMax(val, mask);

  return val;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
