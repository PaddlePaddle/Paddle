// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMON_CUH
#define COMMON_CUH

#include "cublas_v2.h"
#include <cub/cub.cuh>

using kv_float = cub::KeyValuePair<float, float>;
using kv_half = cub::KeyValuePair<half, half>;
using kv_half2 = cub::KeyValuePair<half2, half2>;

template <typename T>
__device__ inline T rsqrt(const T& x);

template <>
__device__ inline float rsqrt(const float& x) {
    return rsqrtf(x);
}

__device__ inline kv_float operator+(const kv_float& a, const kv_float& b) {
    return kv_float(a.key + b.key, a.value + b.value);
}

// Half Operations
__device__ inline half2 __hadd2_with_fallback(const half2 a, const half2 b) {
#if __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 out {};
    out.x = __half2float(a.x) + __half2float(b.x);
    out.y = __half2float(a.y) + __half2float(b.y);
    return __float22half2_rn(out);
#endif
}
#if __CUDA_ARCH__ < 530
template <typename T>
__device__ inline T operator+(const T& a, const T& b);
template <typename T>
__device__ inline T operator*(const T& a, const T& b);
template <>
__device__ inline half2 operator+(const half2& a, const half2& b) {
    return __hadd2_with_fallback(a, b);
}
template <>
__device__ inline half2 operator*(const half2& a, const half2& b) {
    float2 out {};
    out.x = __half2float(a.x) * __half2float(b.x);
    out.y = __half2float(a.y) * __half2float(b.y);
    return __float22half2_rn(out);
}
template <typename T>
__device__ inline T operator+(const T& a, const T& b);
template <typename T>
__device__ inline T operator/(const T& a, const T& b);
template <typename T>
__device__ inline T& operator+=(T& a, const T& b);
template <typename T>
__device__ inline T operator-(const T& a, const T& b);
template <typename T>
__device__ inline T operator*(const T& a, const T& b);
template <>
__device__ inline half operator+(const half& a, const half& b) {
    return __float2half(__half2float(a) + __half2float(b));
}
template <>
__device__ inline half& operator+=(half& a, const half& b) {
    a = __float2half(__half2float(a) + __half2float(b));
    return a;
}
template <>
__device__ inline half operator-(const half& a, const half& b) {
    return __float2half(__half2float(a) - __half2float(b));
}
template <>
__device__ inline half operator*(const half& a, const half& b) {
    return __float2half(__half2float(a) * __half2float(b));
}
template <>
__device__ inline half operator/(const half& a, const half& b) {
    return __float2half(__half2float(a) / __half2float(b));
}
#endif

template <>
__device__ inline half rsqrt(const half& x) {
#if __CUDA_ARCH__ >= 530
    return hrsqrt(x);
#else
    return __float2half(rsqrt(__half2float(x)));
#endif
}

__device__ inline kv_half operator+(const kv_half& a, const kv_half& b) {
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = __hadd2_with_fallback(a2, b2);
    return kv_half(res.x, res.y);
}

__device__ inline kv_half2 operator+(const kv_half2& a, const kv_half2& b) {
    return kv_half2(__hadd2_with_fallback(a.key, b.key), __hadd2_with_fallback(a.value, b.value));
}
// Helper Functions
template <typename T>
using kvp = cub::KeyValuePair<T, T>;
template <typename T, typename R, typename P, int TPB>
__device__ inline void layerNorm(
    const kvp<R>& threadData, const int ld, const int offset, const P* beta, const P* gamma, T* output) {
    // Assuming threadData is already divided by ld
    using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ R mu;     // mean
    __shared__ R rsigma; // 1 / std.dev.
    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());
    if (threadIdx.x == 0) {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < ld; i += TPB) {
        const int idx = offset + i;
        const R val = output[idx];
        const R g(gamma[i]);
        const R b(beta[i]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}

#endif // #ifndef COMMON_CUH
