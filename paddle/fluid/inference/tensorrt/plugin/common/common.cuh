/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef COMMON_CUH
#define COMMON_CUH

#include "cublas_v2.h"
#include <cub/cub.cuh>

#define HDI inline __host__ __device__

using kv_float = cub::KeyValuePair<float, float>;
using kv_half = cub::KeyValuePair<half, half>;
using kv_half2 = cub::KeyValuePair<half2, half2>;

template <typename T>
__device__ inline T tanh(const T& x);

template <typename T>
__device__ inline T rsqrt(const T& x);

template <typename T>
__device__ inline T exp(const T& x);


// Float32 Operations
template <>
__device__ inline float tanh(const float& x)
{
    return tanhf(x);
}

template <>
__device__ inline float rsqrt(const float& x)
{
    return rsqrtf(x);
}

template <>
__device__ inline float exp(const float& x)
{
    return expf(x);
}


__device__ inline kv_float operator+(const kv_float& a, const kv_float& b)
{
    return kv_float(a.key + b.key, a.value + b.value);
}


// Half Operations

__device__ inline half2 __hadd2_with_fallback(const half2 a, const half2 b)
{
#if __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 out{};
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
__device__ inline half2 operator+(const half2& a, const half2& b)
{
    return __hadd2_with_fallback(a, b);
}

template <>
__device__ inline half2 operator*(const half2& a, const half2& b)
{
    float2 out{};
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
__device__ inline half operator+(const half& a, const half& b)
{
    return __float2half(__half2float(a) + __half2float(b));
}

template <>
__device__ inline half& operator+=(half& a, const half& b)
{
    a = __float2half(__half2float(a) + __half2float(b));
    return a;
}

template <>
__device__ inline half operator-(const half& a, const half& b)
{
    return __float2half(__half2float(a) - __half2float(b));
}

template <>
__device__ inline half operator*(const half& a, const half& b)
{
    return __float2half(__half2float(a) * __half2float(b));
}

template <>
__device__ inline half operator/(const half& a, const half& b)
{
    return __float2half(__half2float(a) / __half2float(b));
}

#endif

template <>
__device__ inline half tanh(const half& x)
{
    const float tmp = tanhf(__half2float(x));
    return __float2half(tmp);
}

template <>
__device__ inline half2 tanh(const half2& x)
{
    // at the moment, there is no half2 tanh builtin
    float2 tmp = (__half22float2(x));
    tmp.x = tanhf(tmp.x);
    tmp.y = tanhf(tmp.y);
    return __float22half2_rn(tmp);
}

template <>
__device__ inline half rsqrt(const half& x)
{
#if __CUDA_ARCH__ >= 530
    return hrsqrt(x);
#else
    return __float2half(rsqrt(__half2float(x)));
#endif
}

template <>
__device__ inline half exp(const half& x)
{
#if __CUDA_ARCH__ >= 530
    return hexp(x);
#else
    return __float2half(exp(__half2float(x)));
#endif
}

__device__ inline kv_half operator+(const kv_half& a, const kv_half& b)
{
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = __hadd2_with_fallback(a2, b2);
    return kv_half(res.x, res.y);
}

__device__ inline kv_half2 operator+(const kv_half2& a, const kv_half2& b)
{
    return kv_half2(__hadd2_with_fallback(a.key, b.key), __hadd2_with_fallback(a.value, b.value));
}


// Helper Functions

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T, typename R, typename P, int TPB>
__device__ inline void layerNorm(
    const kvp<R>& threadData, const int ld, const int offset, const P* beta, const P* gamma, T* output)
{
    // Assuming threadData is already divided by ld

    using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ R mu;     // mean
    __shared__ R rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const R val = output[idx];
        const R g(gamma[i]);
        const R b(beta[i]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}

template <typename T, typename P, int TPB>
__device__ inline void layerNormSmall(
    const T val, const kvp<T>& threadData, const int ld, const int idx, const P* beta, const P* gamma, T* output)
{
    // Assuming threadData is already divided by ld
    // Small settings: the block covers the leading dimension TPB >= ld. The input
    // value is available in a register

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        const T g(gamma[threadIdx.x]);
        const T b(beta[threadIdx.x]);
        output[idx] = g * (val - mu) * rsigma + b;
    }
}

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmaxSmall(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{

    using BlockReduce = cub::BlockReduce<float, TPB>;

    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;
    __shared__ float fMax;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(-FLT_MAX);

    const int idx = offset + threadIdx.x;
    if (threadIdx.x < lastValid)
    {
        threadData = input[idx];
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        fMax = maxElem;
    }
    __syncthreads();

    if (threadIdx.x < lastValid)
    {
        threadData = exp((threadData - fMax) * w);
    }
    else
    {
        threadData = 0;
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = (1.f) / Z;
    }
    __syncthreads();

    if (threadIdx.x < ld)
    {
        // this will be 0 for threadIdx.x >= lastValid
        output[idx] = T(threadData * rZ);
    }
}

template <typename T, unsigned TPB>
__device__ inline void scaledSoftmax(
    const int ld, const int lastValid, const float rsqrtHeadSize, const T* input, T* output)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float rZ;
    __shared__ float fMax;

    const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * ld;

    const float w(rsqrtHeadSize);
    cub::Sum sum;
    float threadData(-FLT_MAX);

    if (lastValid >= blockDim.x)
    {
        threadData = 0;
    }
    for (int i = threadIdx.x; i < lastValid; i += TPB)
    {
        const int idx = offset + i;
        threadData = max(static_cast<float>(input[idx]), threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        fMax = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int i = threadIdx.x; i < lastValid; i += TPB)
    {
        const int idx = offset + i;
        threadData += exp((static_cast<float>(input[idx]) - fMax) * w);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        rZ = 1.f / Z;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const float val = (i < lastValid) ? exp((static_cast<float>(input[idx]) - fMax) * w) * rZ : 0.f;
        output[idx] = T(val);
    }
}

template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
    return (a + b - 1) / b;
}
template <typename IntType>
constexpr HDI IntType alignTo(IntType a, IntType b)
{
    return ceildiv(a, b) * b;
}

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template <typename T>
__device__ inline T myExp(const T x);

template <>
__device__ inline half myExp<half>(const half x)
{
    return exp(x);
}
template <>
__device__ inline float myExp<float>(const float x)
{
    return __expf(x);
}

static inline __device__ uint32_t float4_to_char4(float x,
                                                  float y,
                                                  float z,
                                                  float w) {
  uint32_t dst;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  uint32_t a; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(a) : "f"(x));
  uint32_t b; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(b) : "f"(y));
  uint32_t c; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(c) : "f"(z));
  uint32_t d; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(d) : "f"(w));

  asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2,  0;\n" : "=r"(dst) : "r"(d), "r"(c));
  asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %0;\n" : "+r"(dst) : "r"(b), "r"(a));
#else
  char4 tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  tmp.w = w;
  dst = reinterpret_cast<const uint32_t&>(tmp);
#endif
  return dst;
}

inline __device__ char quantize(const float x, const float qScale)
{
    int tmpq = __float2int_rn(qScale * x);  // scale and round
    char tmpq8 = min(127, max(-127, tmpq)); // clip and cast
    return tmpq8;
}

inline __device__ void ldg(const int8_t* input, uint4& data)
{
    data = *reinterpret_cast<const uint4*>(input);
}

inline __device__ void stg(int8_t* output, uint4& data)
{
    *reinterpret_cast<uint4*>(output) = data;
}

inline __device__ uint32_t pack4(const float (&hdata)[4], const float qScale)
{
    return float4_to_char4(hdata[0] * qScale, hdata[1] * qScale , hdata[2] * qScale, hdata[3] * qScale);
}


#endif // #ifndef COMMON_CUH
