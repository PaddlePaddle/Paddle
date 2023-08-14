// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#endif
#include <stdint.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <string>

#include "paddle/phi/core/dense_tensor.h"

namespace phi {
namespace fusion {

#ifdef PADDLE_WITH_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

#define MASK 0xffffffff

__device__ __inline__ void load_data_upper_tri(phi::float16* dst,
                                               const phi::float16* src) {
  *(reinterpret_cast<float2*>(dst)) = *(reinterpret_cast<const float2*>(src));
}

__device__ __inline__ void load_data_upper_tri(phi::bfloat16* dst,
                                               const phi::bfloat16* src) {
  *(reinterpret_cast<float2*>(dst)) = *(reinterpret_cast<const float2*>(src));
}

__device__ __inline__ void load_data_upper_tri(float* dst, const float* src) {
  *(reinterpret_cast<float4*>(dst)) = *(reinterpret_cast<const float4*>(src));
}

__device__ __inline__ void load_zero_vector_upper_tri(phi::float16* dst) {
  *(reinterpret_cast<float2*>(dst)) = make_float2(0.0f, 0.0f);
}

__device__ __inline__ void load_zero_vector_upper_tri(phi::bfloat16* dst) {
  *(reinterpret_cast<float2*>(dst)) = make_float2(0.0f, 0.0f);
}

__device__ __inline__ void load_zero_vector_upper_tri(float* dst) {
  *(reinterpret_cast<float4*>(dst)) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__inline__ int get_pow2_index_value(int value) {
  int pow2_index = 0;
  while ((1 << pow2_index) < value) {
    ++pow2_index;
  }
  return pow2_index;
}

template <typename T>
struct AddOP_upper_tri {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct MaxOP_upper_tri {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T warp_shfl_xor_upper_tri(T value,
                                                     int laneMask,
                                                     int width,
                                                     unsigned int mask = MASK) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T, int batch, int width, template <typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce_upper_tri(T* sum) {
  ReduceOp<T> r;
#pragma unroll
  for (int offset = width / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < batch; ++i) {
      T b = warp_shfl_xor_upper_tri(sum[i], offset, width);
      sum[i] = r(sum[i], b);
    }
  }
}
}  // namespace fusion
}  // namespace phi
