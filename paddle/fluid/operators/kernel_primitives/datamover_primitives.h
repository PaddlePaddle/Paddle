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

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

#define INT_BITS 32

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

template <int kDims>
struct BroadcastConfig {
  FastDivMod divmoders[kDims];
  uint32_t strides[framework::DDim::kMaxRank];
  HOSTDEVICE BroadcastConfig() {}

  HOSTDEVICE BroadcastConfig(const std::vector<int64_t>& out_dims,
                             const std::vector<int64_t>& in_dims,
                             int dim_size) {
    std::vector<uint32_t> strides_in;
    std::vector<FastDivMod> divmoders_in;
    // for divmoders
    divmoders_in.resize(dim_size);
    for (int i = 0; i < dim_size; ++i) {
      divmoders_in[i] = FastDivMod(out_dims[i]);
    }
    // for strides
    strides_in.resize(dim_size, 1);
    for (int i = 0; i < dim_size; ++i) {
      strides_in[i] = in_dims[i] == 1 ? 0 : strides_in[i];
      strides_in[i] =
          (i != 0 && strides_in[i] != 0)
              ? std::accumulate(in_dims.begin(), in_dims.begin() + i, 1,
                                std::multiplies<int64_t>())
              : strides_in[i];
    }

    memcpy(strides, strides_in.data(), kDims * sizeof(uint32_t));
    memcpy(divmoders, divmoders_in.data(), kDims * sizeof(FastDivMod));
  }
};

#undef INT_BITS
}  // namespace details

template <typename T, int NX, int NY, int BlockSize>
__device__ __forceinline__ void ReadDataBase(T* dst, const T* __restrict__ src,
                                             int size) {
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
__device__ __forceinline__ void ReadData(T* dst, const T* __restrict__ src,
                                         int size) {
  const int VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
  const int VECTORS_PER_THREAD = NX / VECTOR_SIZE;

  // Vector per thread
  if (blockDim.x * NX > size) {
    ReadDataBase<T, NX, NY, BlockSize>(dst, src, size);
  } else {
    // Vector type
    using VecType = details::VectorType<T, VECTOR_SIZE>;
    VecType vec_temp[VECTORS_PER_THREAD];
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    ReadDataBase<VecType, VECTORS_PER_THREAD, NY, BlockSize>(
        vec_temp, vec_input, VECTORS_PER_THREAD * blockDim.x);
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
    }
  }
}

/** @brief: ReadDataBc
 * read data from src ptr when the shape of src and dst are different
 * @param：
 * src: the source pointer
 * dst: the dst pointer
 * stride_nx: the stride of src
 * stride_ny: the stride of src
 * the shape of dst is [NY, NX]
 */
template <typename T, int NX, int NY, int BlockSize, int ShapeSize>
__device__ __forceinline__ void ReadDataBc(
    T* dst, const T* __restrict__ src, uint32_t fix,
    details::BroadcastConfig<ShapeSize> config, int num, int stride_nx,
    int stride_ny) {
  uint32_t base_offset = fix + threadIdx.x * NX;
  uint32_t offset = 0;

#pragma unroll
  for (int ny = 0; ny < NY; ++ny) {
#pragma unroll
    for (uint32_t nx = 0; nx < NX; ++nx) {
      uint32_t idx = base_offset + ny * stride_ny + nx * stride_nx;
      if (idx < num) {
        offset = 0;
#pragma unroll
        for (int i = 0; i < ShapeSize; ++i) {
          auto fast_divmoder = config.divmoders[i].Divmod(idx);
          idx = fast_divmoder.val[0];
          offset += fast_divmoder.val[1] * config.strides[i];
        }
        dst[nx + ny * NX] = src[offset];
      }
    }
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ __forceinline__ void WriteDataBase(T* dst, const T* __restrict__ src,
                                              int size) {
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
__device__ __forceinline__ void WriteData(T* dst, T* __restrict__ src,
                                          int size) {
  const int VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
  const int VECTORS_PER_THREAD = NX / VECTOR_SIZE;

  // Vector per thread
  if (blockDim.x * NX > size) {
    WriteDataBase<T, NX, NY, BlockSize>(dst, src, size);
  } else {
    // Vector type
    using VecType = details::VectorType<T, VECTOR_SIZE>;
    VecType vec_temp[VECTORS_PER_THREAD];
#pragma unroll
    for (int idx = 0; idx < VECTORS_PER_THREAD; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType*>(src) + idx);
    }
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    WriteDataBase<VecType, VECTORS_PER_THREAD, NY, BlockSize>(
        vec_dst, vec_temp, VECTORS_PER_THREAD * blockDim.x);
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
