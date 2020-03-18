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

#ifdef SUPPORT_CUDA_FP16
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

#ifdef SUPPORT_CUDA_FP16
template <>
__device__ __forceinline__ float ToFloat<half>(half a) {
  return __half2float(a);
}
#endif

template <>
__device__ __forceinline__ float exp_func<float>(float a) {
  return expf(a);
}

#ifdef SUPPORT_CUDA_FP16
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

#ifdef SUPPORT_CUDA_FP16
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
