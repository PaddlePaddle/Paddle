// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <iostream>
#include <vector>

#define INT_BITS 32

namespace paddle {
namespace operators {
namespace kernel_primitives {

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

struct FastDivMod {
  // 1st value represents the result of input number divides by recorded divisor
  // 2nd value represents the result of input number modulo by recorded divisor
  using DivModT = VectorType<uint32_t, 2>;

  FastDivMod() {}
  HOSTDEVICE FastDivMod(uint32_t d) : divisor(d) {
    static_assert(sizeof(unsigned int) == 4,
                  "Only Support 32-bit unsigned int.");

    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      auto shift_limit = 1 << shift_val;
      if (shift_limit >= divisor) break;
    }
    uint64_t long_one = 1;
    uint64_t temp_div =
        ((long_one << INT_BITS) * ((long_one << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
    uint32_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint32_t n) const {
    uint32_t q = Div(n);
    DivModT result = {q, n - q * divisor};
    return result;
  }

  int32_t divisor;
  int32_t shift_val;
  uint32_t multiplier;
};

template <typename T, int NX, int NY, int BlockSize>
__device__ void read_data_base(T* dst, const T* __restrict__ src, int size) {
  int dx = threadIdx.x * NX;
#pragma unroll 4
  for (int idx = 0; idx < NX; ++idx) {
    if ((idx + dx) >= size) {
      break;
    }
    dst[idx] = src[idx + dx];
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void read_data(T* dst, const T* __restrict__ src, int size) {
  const int VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
  const int VECTORS_PER_THREAD = NX / VECTOR_SIZE;

  // Vector per thread
  if (blockDim.x * NX > size) {
    read_data_base<T, NX, NY, BlockSize>(dst, src, size);
  } else {
    // Vector type
    using VecType = VectorType<T, VECTOR_SIZE>;
    VecType vec_temp[VECTORS_PER_THREAD];
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    read_data_base<VecType, VECTORS_PER_THREAD, NY, BlockSize>(
        vec_temp, vec_input, VECTORS_PER_THREAD * blockDim.x);
#pragma unroll 2
    for (int idx = 0; idx < NX; ++idx) {
      dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
    }
  }
}

/** @brief: read_data_bc
 * read data from src ptr when the shape of src and dst are different
 * @param：
 * src: the source pointer
 * dst: the dst pointer
 * stride_nx: the stride of src
 * stride_ny: the stride of src
 * the shape of dst is [NY, NX]
 */
template <typename T, int NX, int NY, int BS, int ShapeSize>
__device__ __forceinline__ void read_data_bc(T* dst, const T* __restrict__ src,
                                             uint32_t fix,
                                             FastDivMod* divmoders,
                                             uint32_t* strides, int stride_nx,
                                             int stride_ny) {
  uint32_t base_offset = fix + threadIdx.x * NX;
  uint32_t offset = 0;

#pragma unroll 2
  for (int ny = 0; ny < NY; ++ny) {
#pragma unroll 4
    for (uint32_t nx = 0; nx < NX; ++nx) {
      uint32_t idx = base_offset + ny * stride_ny + nx * stride_nx;
      offset = 0;
#pragma unroll 2
      for (int i = 0; i < ShapeSize; ++i) {
        auto fast_divmoder = divmoders[i].Divmod(idx);
        idx = fast_divmoder.val[0];
        offset += fast_divmoder.val[1] * strides[i];
      }
      dst[nx + ny * NX] = src[offset];
    }
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void write_data_base(T* dst, const T* __restrict__ src, int size) {
  int dx = threadIdx.x * NX;
#pragma unroll 4
  for (int idx = 0; idx < NX; ++idx) {
    if ((idx + dx) >= size) {
      break;
    }
    dst[idx + dx] = src[idx];
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void write_data(T* dst, T* __restrict__ src, int size) {
  const int VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
  const int VECTORS_PER_THREAD = NX / VECTOR_SIZE;

  // Vector per thread
  if (blockDim.x * NX > size) {
    write_data_base<T, NX, NY, BlockSize>(dst, src, size);
  } else {
    // Vector type
    using VecType = VectorType<T, VECTOR_SIZE>;
    VecType vec_temp[VECTORS_PER_THREAD];
#pragma unroll 2
    for (int idx = 0; idx < VECTORS_PER_THREAD; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType*>(src) + idx);
    }
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    write_data_base<VecType, VECTORS_PER_THREAD, NY, BlockSize>(
        vec_dst, vec_temp, VECTORS_PER_THREAD * blockDim.x);
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
