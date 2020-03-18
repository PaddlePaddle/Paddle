// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ T FromFloat(float a);

template <typename T>
__device__ __forceinline__ float ToFloat(T a);

template <typename T>
__device__ __forceinline__ T exp_func(T a);

// from_float
template <>
__device__ __forceinline__ float FromFloat<float>(float a) {
  return a;
}

template <>
__device__ __forceinline__ half FromFloat<half>(float a) {
  return __float2half(a);
}

// to_float
template <>
__device__ __forceinline__ float ToFloat<float>(float a) {
  return a;
}

template <>
__device__ __forceinline__ float ToFloat<half>(half a) {
  return __half2float(a);
}

template <>
__device__ __forceinline__ float exp_func<float>(float a) {
  return expf(a);
}

template <>
__device__ __forceinline__ half exp_func<half>(half a) {
#if __CUDA_ARCH__ >= 600
  return hexp(a);
#else
  return FromFloat<half>(expf(ToFloat<half>(a)));
#endif
}
