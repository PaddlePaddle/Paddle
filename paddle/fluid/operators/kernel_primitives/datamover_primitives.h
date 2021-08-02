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
#include "paddle/fluid/platform/fast_divmod.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

template <typename T, int NX, int NY, int BlockSize>
__device__ void read_data_base(T* dst, const T* __restrict__ src, int size) {
  int dx = threadIdx.x * NX;
#pragma unroll
  for (int idx = 0; idx < NX; ++idx) {
    if ((idx + dx) >= size) {
      break;
    }
    dst[idx] = src[idx + dx];
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void read_data(T* dst, const T* __restrict__ src, int size) {
  enum {
    VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1,
    VECTORS_PER_THREAD = NX / VECTOR_SIZE,
  };

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
#pragma unroll
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
template <typename T, int NX, int NY, int BS, int Shape_Size>
__device__ __forceinline__ void read_data_bc(
    T* dst, const T* __restrict__ src, uint32_t fix, FastDivMod* divmoders,
    uint32_t* strides, uint32_t stride_nx, uint32_t stride_ny) {
  uint32_t base_offset = fix + threadIdx.x * NX;
  uint32_t offset = 0;

#pragma unroll
  for (int ny = 0; ny < NY; ++ny) {
#pragma unroll
    for (uint32_t nx = 0; nx < NX; ++nx) {
      uint32_t idx = base_offset + ny * stride_ny + nx * stride_nx;
      offset = 0;
#pragma unroll
      for (int i = 0; i < Shape_Size; ++i) {
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
#pragma unroll
  for (int idx = 0; idx < NX; ++idx) {
    if ((idx + dx) >= size) {
      break;
    }
    dst[idx + dx] = src[idx];
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void write_data(T* dst, T* __restrict__ src, int size) {
  enum {
    VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1,
    VECTORS_PER_THREAD = NX / VECTOR_SIZE,
  };

  // Vector per thread
  if (blockDim.x * NX > size) {
    write_data_base<T, NX, NY, BlockSize>(dst, src, size);
  } else {
    // Vector type
    using VecType = VectorType<T, VECTOR_SIZE>;
    VecType vec_temp[VECTORS_PER_THREAD];
#pragma unroll
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
