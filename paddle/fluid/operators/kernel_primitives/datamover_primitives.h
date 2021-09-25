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

/**
 * @brief load data from src to dst, src can be 1D data or 2D data. Note that
 * you can use this function when you are sure that the data will not cross the
 * boundary.
 * @typename:
 * Tx: data type of src
 * Ty: data type of dstt
 * NX: the cols of src, dst
 * NY: the rows of src, dst
 * BlockSize: the config of this device
 * @param：
 * stride_nx: the stride of cols
 * stride_ny: the stride of rows
 */

template <typename Tx, typename Ty, int NX, int NY, int BlockSize>
__device__ __forceinline__ void ReadData(Ty* dst, const Tx* __restrict__ src,
                                         int stride_nx, int stride_ny) {
  int thread_offset = threadIdx.x * NX;

  if (NY == 1 && NX == 1) {
    dst[0] = static_cast<Ty>(src[thread_offset]);
  } else if (NX == 1) {
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      dst[idy] = static_cast<Ty>(src[thread_offset + idy * stride_ny]);
    }
  } else if (NY == 1) {
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      dst[idx] = static_cast<Ty>(src[thread_offset + idx * stride_nx]);
    }
  } else {
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        dst[idy * NX + idx] = static_cast<Ty>(
            src[thread_offset + idx * stride_nx + idy * stride_ny]);
      }
    }
  }
}

/**
 * @brief load data from src to dst with stride, src can be 1D data or 2D data.
 * When boundary judgment is required, you need to set a to true, and a is false
 * by default.
 * @typename:
 * Tx: data type of src
 * Ty: data type of dstt
 * NX: the cols of src, dst
 * NY: the rows of src, dst
 * BlockSize: the config of this device
 * IsBoundary: whether to make boundary judgment
 * @param：
 * size_nx: number of columns to be processed by the current block
 * size_ny: number of rows to be processed by the current block
 * stride_nx: the stride of cols
 * stride_ny: the stride of rows
 */
template <typename Tx, typename Ty, int NX, int NY, int BlockSize,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadData(Ty* dst, const Tx* __restrict__ src,
                                         int size_nx, int size_ny,
                                         int stride_nx, int stride_ny) {
  int thread_offset = threadIdx.x * NX;
  int left_size_nx = size_nx - thread_offset;

  // Each branch is added for better performance
  if (NX == 1 && NY == 1) {  // for NX == 1 and NY == 1
    if (IsBoundary) {
      if (left_size_nx > 0) {
        dst[0] = static_cast<Ty>(src[thread_offset]);
      }
    } else {
      dst[0] = static_cast<Ty>(src[thread_offset]);
    }
  } else if (NX == 1) {  // for NX == 1 and NY != 1
#pragma unroll
    for (int idy = 0; idy < NY; ++idy) {
      if (IsBoundary) {
        if (idy >= size_ny) {
          break;
        }
      }
      dst[idy] = static_cast<Ty>(src[thread_offset + idy * stride_ny]);
    }
  } else if (NY == 1) {  // for NY == 1 and NX != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx >= left_size_nx) {
          break;
        }
      }
      dst[idx] = static_cast<Ty>(src[thread_offset + idx * stride_nx]);
    }
  } else {  // for NX != 1 and NY != 1
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (IsBoundary) {
        if (idx >= left_size_nx) {
          break;
        }
      }
#pragma unroll
      for (int idy = 0; idy < NY; ++idy) {
        if (IsBoundary) {
          if (idy >= size_ny) {
            break;
          }
        }
        dst[idy * NX + idx] = static_cast<Ty>(
            src[thread_offset + idx * stride_nx + idy * stride_ny]);
      }
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

/** @brief: ReadData
 * @brief load data from src to dst, src can be 1D data, you should set NY = 1.
 * When boundary judgment is required, you need to set a to true, and a is false
 * by default.
 * @typename:
 * T : the data type of src
 * NX: the cols of src, dst
 * NY: in this function NY only can be 1
 * BlockSize: the config of this device
 * IsBoundary: whether to make boundary judgment
 * @param：
 * num: number of columns to be processed by the current block
 */
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ __forceinline__ void ReadData(T* dst, const T* __restrict__ src,
                                         int num) {
  if (IsBoundary) {  // blockDim.x * NX > num
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if (idx + thread_offset < num) {
        dst[idx] = src[thread_offset + idx];
      }
    }
  } else {  // blockDim,x * NX < num
    const int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    const int kVectorsPerThread = NX / kVectorSize;
    int thread_offset = threadIdx.x * kVectorsPerThread;

    using VecType = details::VectorType<T, kVectorSize>;
    const VecType* vec_input = reinterpret_cast<const VecType*>(src);
    VecType vec_temp[kVectorsPerThread];

#pragma unroll
    for (int i = 0; i < kVectorsPerThread; ++i) {
      vec_temp[i] = vec_input[thread_offset + i];
#pragma unroll
      for (int idx = 0; idx < NX; ++idx) {
        dst[idx] = *(reinterpret_cast<T*>(vec_temp) + idx);
      }
    }
  }
}

/**
 * @brief: read data for broadcast
 * @typename:
 * T : the data type of src
 * NX: the cols of src, dst
 * NY: in this function NY only can be 1
 * BlockSize: the config of this device
 * ShapeSize: the shape size of out. eg in[1, 35], out[32, 35] then shape size
 * is 2
 * IsBoundary: whether to make boundary judgment
 * @param：
 * block_offset: data offset of this block, blockDim.x * blockIdx.x * NX;
 * config: get the global index in src, attention config was declared in host;
 * total_num_output: total num of output
 * stride_nx: the stride of cols
 * stride_ny: the stride of rows
 */
template <typename T, int NX, int NY, int BlockSize, int ShapeSize,
          bool IsBoundary = false>
__device__ __forceinline__ void ReadDataBc(
    T* dst, const T* __restrict__ src, uint32_t block_offset,
    details::BroadcastConfig<ShapeSize> config, int total_num_output,
    int stride_nx, int stride_ny) {
  uint32_t thread_offset = block_offset + threadIdx.x * NX;
  uint32_t index_src = 0;

#pragma unroll
  for (int ny = 0; ny < NY; ++ny) {
#pragma unroll
    for (uint32_t nx = 0; nx < NX; ++nx) {
      uint32_t index_output = thread_offset + ny * stride_ny + nx * stride_nx;
      index_src = 0;
      if (IsBoundary) {
        if (index_output >= total_num_output) {
          break;
        }
      }
#pragma unroll
      for (int i = 0; i < ShapeSize; ++i) {
        auto fast_divmoder = config.divmoders[i].Divmod(index_output);
        index_output = fast_divmoder.val[0];
        index_src += fast_divmoder.val[1] * config.strides[i];
      }
      dst[nx + ny * NX] = src[index_src];
    }
  }
}

/**
 * @brief: read data for broadcast
 * @typename:
 * T : the data type of src
 * NX: the cols of src, dst
 * NY: in this function NY only can be 1
 * BlockSize: the config of this device
 * ShapeSize: the shape size of out. eg in[1, 35], out[32, 35] then shape size
 * is 2
 * IndexCal: get the global index in src, attention config was declared in host;
 * IsBoundary: whether to make boundary judgment
 * @param：
 * block_offset: data offset of this block, blockDim.x * blockIdx.x * NX;
 * index_cal: get the global index in src, attention config was declared in
 * host;
 * size_nx: number of columns to be processed by the current block
 * size_ny: number of rows to be processed by the current block
 * stride_nx: the stride of cols
 * stride_ny: the stride of rows
 * reduce_last_dim: according to the block split set threadIdx
 */
template <typename T, int NX, int NY, int BlockSize, int ShapeSize,
          typename IndexCal, bool IsBoundary = false>
__device__ __forceinline__ void ReadDataReduce(
    T* dst, const T* __restrict__ src, int block_offset,
    const IndexCal& index_cal, int size_nx, int size_ny, int stride_nx,
    int stride_ny, bool reduce_last_dim) {
  int thread_offset = 0;
  if (reduce_last_dim) {
    thread_offset = block_offset + threadIdx.x;
  } else {
    thread_offset = block_offset + threadIdx.y;
  }

  if (NX == 1) {
#pragma unroll
    for (int ny = 0; ny < NY; ++ny) {
      if (IsBoundary) {
        if (thread_offset >= size_ny) {
          break;
        }
      }
      uint32_t index_src = index_cal(thread_offset);
      dst[ny] = src[index_src];
      thread_offset += stride_ny;
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
        uint32_t index_src = index_cal(thread_offset);
        dst[nx + ny * NX] = src[index_src];
        thread_offset += stride_ny;
      }
      thread_offset += stride_nx;
    }
  }
}

/**
 * @brief store data from src to dst, src can be 1D data, you should set NY = 1.
 * When boundary judgment is required, you need to set a to true, and a is false
 * by default.
 * @typename:
 * T : the data type of src
 * NX: the cols of src, dst
 * NY: in this function NY only can be 1
 * BlockSize: the config of this device
 * IsBoundary: whether to make boundary judgment
 * @param：
 * num: number of columns to be processed by the current block
 */
template <typename T, int NX, int NY, int BlockSize, bool IsBoundary = false>
__device__ __forceinline__ void WriteData(T* dst, T* __restrict__ src,
                                          int num) {
  if (IsBoundary) {
    int thread_offset = threadIdx.x * NX;
#pragma unroll
    for (int idx = 0; idx < NX; ++idx) {
      if ((thread_offset + idx) < num) {
        dst[thread_offset + idx] = src[idx];
      }
    }
  } else {
    // Vector type
    const int kVectorSize = (NX % 4 == 0) ? 4 : (NX % 2 == 0) ? 2 : 1;
    const int kVectorsPerThread = NX / kVectorSize;

    int thread_offset = threadIdx.x * kVectorsPerThread;
    using VecType = details::VectorType<T, kVectorSize>;
    VecType* vec_dst = reinterpret_cast<VecType*>(dst);
    VecType vec_temp[kVectorsPerThread];
#pragma unroll
    for (int idx = 0; idx < kVectorsPerThread; ++idx) {
      vec_temp[idx] = *(reinterpret_cast<VecType*>(src) + idx);
      vec_dst[thread_offset + idx] = vec_temp[idx];
    }
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
