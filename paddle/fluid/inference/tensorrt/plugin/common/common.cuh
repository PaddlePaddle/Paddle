// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
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

#include <cub/cub.cuh>
#include "cublas_v2.h"
#include "paddle/fluid/platform/device_context.h"
namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

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
__device__ inline half2 operator+(const half2& a, const half2& b) {
  return __hadd2_with_fallback(a, b);
}
template <>
__device__ inline half2 operator*(const half2& a, const half2& b) {
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
  return kv_half2(__hadd2_with_fallback(a.key, b.key),
                  __hadd2_with_fallback(a.value, b.value));
}
// Helper Functions
template <typename T>
using kvp = cub::KeyValuePair<T, T>;
template <typename T, typename R, typename P, int TPB>
__device__ inline void layerNorm(const kvp<R>& threadData,
                                 const int ld,
                                 const int offset,
                                 const P* beta,
                                 const P* gamma,
                                 T* output) {
  // Assuming threadData is already divided by ld
  using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ R mu;      // mean
  __shared__ R rsigma;  // 1 / std.dev.
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

// Helper Functions for multihead related plugins
template <typename T>
__global__ void transpose(T *src,
                          T *dst,
                          const int batch_size,
                          const int seq_len,
                          const int head_num,
                          const int size_per_head) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) +
      seq_id * head_num * size_per_head + head_id * size_per_head +
      threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <typename T>
__global__ void TransposeQkvKernel(const int H, const T *input, T *output) {
  // Input: BxSx3xNxH
  // Bias: 3xSxB
  // Output: 3xBxNxSxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int N = blockDim.y;
  const int S = gridDim.x;
  const int B = gridDim.y;

  const int NH = N * H;
  const int NHS = NH * S;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * S * H + b * NHS + m * NHS * B;

  const int i = threadIdx.x;
  output[out_offset + i] = input[in_offset + i];
}

inline void TransposeQKV(const int batch,
                         const int seq_len,
                         const int head_size,
                         const int head_num,
                         const float *input,
                         float *output,
                         cudaStream_t stream) {
  int scratch_size = batch * head_num * seq_len * seq_len;
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 4 == 0 && scratch_size % 4 == 0) {
    const int h = head_size / 4;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);
    float4 *output4 = reinterpret_cast<float4 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 4));
    TransposeQkvKernel<float4><<<grid, block, 0, stream>>>(h, input4, output4);
  } else if (head_size % 2 == 0 && scratch_size % 2 == 0) {
    const int h = head_size / 2;
    const float2 *input2 = reinterpret_cast<const float2 *>(input);
    float2 *output2 = reinterpret_cast<float2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 2));
    TransposeQkvKernel<float2><<<grid, block, 0, stream>>>(h, input2, output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024));
    TransposeQkvKernel<float>
        <<<grid, block, 0, stream>>>(head_size, input, output);
  }
}

inline void TransposeQKV(const int batch,
                         const int seq_len,
                         const int head_size,
                         const int head_num,
                         const half *input,
                         half *output,
                         cudaStream_t stream) {
  int scratch_size = batch * head_num * seq_len * seq_len;
  const dim3 grid(seq_len, batch, 3);
  if (head_size % 8 == 0 && scratch_size % 8 == 0) {
    int h = head_size / 8;
    const int4 *input4 = reinterpret_cast<const int4 *>(input);
    int4 *output4 = reinterpret_cast<int4 *>(output);
    dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 8));
    TransposeQkvKernel<int4><<<grid, block, 0, stream>>>(h, input4, output4);
  } else if (head_size % 2 == 0 && scratch_size % 2 == 0) {
    const int h = head_size / 2;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);
    half2 *output2 = reinterpret_cast<half2 *>(output);
    const dim3 block(h, head_num, 1);
    // limit h * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(h * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024 * 2));
    TransposeQkvKernel<half2><<<grid, block, 0, stream>>>(h, input2, output2);
  } else {
    const dim3 block(head_size, head_num, 1);
    // limit head_size * head_num to max block size(1024).
    PADDLE_ENFORCE_LE(head_size * head_num,
                      1024,
                      platform::errors::InvalidArgument(
                          "head_num (%d) * head_size (%d) should <= %d",
                          head_num,
                          head_size,
                          1024));
    TransposeQkvKernel<half>
        <<<grid, block, 0, stream>>>(head_size, input, output);
  }
}
}
}
}
}