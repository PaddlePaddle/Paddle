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
#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#endif
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

// dst[NY][NX];
template <typename Tx, typename Ty, int NX, int NY, int BlockSize>
__device__ __forceinline__ void ReadData(Ty* dst, const Tx* __restrict__ src,
                                         int stride_nx, int stride_ny) {
  if (NY == 1 && NX == 1) {
    dst[0] = static_cast<Ty>(src[threadIdx.x]);
  } else if (NX == 1) {
    int dx = threadIdx.x;
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      dst[idy] = static_cast<Ty>(src[dx + idy * stride_ny]);
    }
  } else if (NY == 1) {
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      dst[idx] = static_cast<Ty>(src[idx * stride_nx]);
    }
  } else {
    int dx = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        dst[idy * NX + idx] =
            static_cast<Ty>(src[idx * stride_nx + dx + idy * stride_ny]);
      }
    }
  }
}

// dst[NY][NX];
template <typename Tx, typename Ty, int NX, int NY, int BlockSize>
__device__ __forceinline__ void ReadData(Ty* dst, const Tx* __restrict__ src,
                                         int size_nx, int size_ny,
                                         int stride_nx, int stride_ny) {
  int dx = threadIdx.x * NX;
  int size = size_nx - dx;
#pragma unroll
  for (int idx = 0; idx < NX; ++idx) {
    if (idx >= size) {
      break;
    }
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (idy >= size_ny) {
        break;
      }
      dst[idy * NX + idx] =
          static_cast<Ty>(src[idx * stride_nx + dx + idy * stride_ny]);
    }
  }
}
template <typename T, int NX>
__device__ __forceinline__ void Init(T* dst, T init_data) {
#pragma unroll
  for (int i = 0; i < NX; i++) {
    dst[i] = init_data;
  }
}

// when shape size is 1, this function can be used
/** @brief: ReadData
 * when shape size is 1, this function can be used, eg: elementwise
 * @param：
 * src: the source pointer
 * dst: the dst pointer
 * size: how many data have to deal with
 * attention : when shape size is 1, dst[NY, NX] will be dst[1, NX * NY]
 *             you should set NY' = 1, NX' = NY * NX
 * @typename:
 * Tx : the data type of src
 * Ty : the data type of dst
 * NX : how many data will be deal with per times and per thread
 * NY : 1
 * BlockSize : the config of device
 * IsBoundary : blockDim.x * NX is larger than num
 */
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T* dst, const T* __restrict__ src,
                                         int num) {
  if (IsBoundary) {  // blockDim.x * NX > num
    int dx = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + dx < num) {
        dst[idx] = src[idx + dx];
      }
    }
  } else {  // blockDim,x * NX < num
    const int VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    const int VECTORS_PER_THREAD = NX / VECTOR_SIZE;
    int tid = threadIdx.x * VECTORS_PER_THREAD;

    using VecType = details::VectorType<T, VECTOR_SIZE>;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    VecType vec_temp[VECTORS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < VECTORS_PER_THREAD; ++i) {
      vec_temp[i] = vec_input[i + tid];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
      }
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
 * @typename:
 * Tx : the data type of src
 * Ty : the data type of dst
 * NX : number of columns to be processed continuously per thread
 * NY : number of rows to be processed per thread
 * BlockSize : the config of device
 * IsBoundary : blockDim.x * NX is larger than num
 */
template <typename T, int NX, int NY, int BlockSize, int ShapeSize,
          bool IsBoundary = false>
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
      if (IsBoundary) {
        if (idx >= num) {
          break;
        }
      }
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

/** @brief: ReadDataReduce
 * read data from global memory for reduce op, the idx of src should be less
 * than size_nx and size_ny
 * @param：
 * src: the source pointer
 * dst: the dst pointer
 * stride_nx: the stride of src
 * stride_ny: the stride of src
 * the shape of dst is [NY, NX]
 */
template <typename T, int NX, int NY, int BlockSize, int ShapeSize,
          typename IndexCal, bool IsBoundary = false>
__device__ __forceinline__ void ReadDataReduce(
    T* dst, const T* __restrict__ src, int fix, const IndexCal& index_cal,
    int size_nx, int size_ny, int stride_nx, int stride_ny,
    bool reduce_lastdim) {
  int base_offset = fix;
  if (reduce_lastdim) {
    base_offset += threadIdx.x;
  } else {
    base_offset += threadIdx.y;
  }

  if (NX == 1) {
#pragma unroll
    for (int ny = 0; ny < NY; ++ny) {
      if (IsBoundary) {
        if (base_offset >= size_ny) {
          break;
        }
      }
      uint32_t offset = index_cal(base_offset);
      dst[ny] = src[offset];
      base_offset += stride_ny;
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < NX; ++nx) {
      if (IsBoundary) {
        if (nx * stride_nx >= size_nx) {
          break;
        }
      }
#pragma unroll
      for (int ny = 0; ny < NY; ++ny) {
        if (IsBoundary) {
          if (nx * stride_nx >= size_nx) {
            break;
          }
        }
        uint32_t offset = index_cal(base_offset);
        dst[nx + ny * NX] = src[offset];
        base_offset += stride_ny;
      }
    }
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ __forceinline__ void WriteData(T* dst, const T* __restrict__ src) {
  int dx = threadIdx.x * NX;
#pragma unroll
  for (int idx = 0; idx < NX; ++idx) {
    dst[idx + dx] = src[idx];
  }
}

template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(T* dst, T* __restrict__ src,
                                          int num) {
  if (IsBoundary) {
    int dx = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if ((idx + dx) < num) {
        dst[idx + dx] = src[idx];
      }
    }
  } else {
    // Vector type
    const int VECTOR_SIZE = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    const int VECTORS_PER_THREAD = NX / VECTOR_SIZE;

    int dx = threadIdx.x * VECTORS_PER_THREAD;
    using VecType = details::VectorType<T, VECTOR_SIZE>;
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    VecType vec_temp[VECTORS_PER_THREAD];
#pragma unroll
    for (int idx = 0; idx < VECTORS_PER_THREAD; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType*>(src) + idx);
      vec_dst[dx + idx] = vec_temp[idx];
    }
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
