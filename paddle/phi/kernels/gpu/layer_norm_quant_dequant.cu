/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Original OneFlow copyright notice:

/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/layer_norm.cuh
// The following code modified from OneFlow's implementation, and change to use
// single Pass algorithm. Support Int8 quant, dequant Load/Store implementation
// for PTQ.

#pragma once

#include "paddle/phi/kernels/layer_norm_quant_dequant.h"
#include <assert.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"
#ifndef PADDLE_WITH_HIP
#include <math_constants.h>
#include <cub/cub.cuh>
#endif

namespace phi {

namespace {

#ifndef PADDLE_WITH_HIP

constexpr int kWarpSize = 32;

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max(a, b);
  }
};

template <template <typename> class ReductionOp,
          typename T,
          int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(
        val, __shfl_xor_sync(0xffffffff, val, mask, thread_group_width));
  }
  return val;
}

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) {
    result_broadcast = result;
  }
  __syncthreads();
  return result_broadcast;
}

template <typename T>
__inline__ __device__ T Div(T a, T b);

template <>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template <>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

template <typename T>
__inline__ __device__ T Rsqrt(T x);

template <>
__inline__ __device__ float Rsqrt<float>(float x) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __frsqrt_rn(x);
#else
  return rsqrt(x);
#endif
}

template <>
__inline__ __device__ double Rsqrt<double>(double x) {
  return rsqrt(x);
}

template <class Func>
inline cudaError_t GetNumBlocks(Func func,
                                int64_t block_size,
                                size_t dynamic_smem_size,
                                int64_t max_blocks,
                                int64_t waves,
                                int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int max_active_blocks;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, func, block_size, dynamic_smem_size);
  }
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template <typename T>
struct DefaultComputeType {
  using type = T;
};

template <>
struct DefaultComputeType<half> {
  using type = float;
};

#if CUDA_VERSION >= 11000
template <>
struct DefaultComputeType<nv_bfloat16> {
  using type = float;
};
#endif  // CUDA_VERSION >= 11000

template <typename T>
class HasCanPackAs {
  typedef char one;
  struct two {
    char x[2];
  };

  template <typename C>
  static one test(decltype(&C::CanPackAs));
  template <typename C>
  static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template <typename T>
typename std::enable_if<HasCanPackAs<T>::value == true, bool>::type CanPackAs(
    T t, size_t pack_size) {
  return t.CanPackAs(pack_size);
}

template <typename T>
typename std::enable_if<HasCanPackAs<T>::value == false, bool>::type CanPackAs(
    T t, size_t pack_size) {
  return true;
}

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
  __device__ Pack() {
    // do nothing
  }
  T elem[N];
};

template <typename SRC, typename DST>
struct DirectLoad {
  using LoadType = DST;
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack = *(reinterpret_cast<const Pack<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }
  const SRC* src;
  int64_t row_size;
};

template <typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template <int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pack.elem[i] = static_cast<DST>(src[i]);
    }
    *(reinterpret_cast<Pack<DST, N>*>(dst) + offset) = pack;
  }
  DST* dst;
  int64_t row_size;
};

template <typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

template <typename T>
inline __device__ void WelfordCombine(
    T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) {
  if (b_count == 0) {
    return;
  }
  T new_count = *count + b_count;
  T nb_over_n = Div(b_count, new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(
    T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(
    T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  WelfordWarpReduce<T, thread_group_width>(
      thread_mean, thread_m2, thread_count, mean, m2, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpReduceSum(T x) {
  T result = 0.0f;
#pragma unroll
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    result += __shfl_xor_sync(0xffffffff, x, mask, thread_group_width);
  }
  return result;
}

template <typename T>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean,
                                                 T thread_m2,
                                                 T thread_count,
                                                 T* result_mean,
                                                 T* result_m2,
                                                 T* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ T count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ T count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0;
  T warp_m2 = 0;
  T warp_count = 0;
  WelfordWarpReduce(
      thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<T>(0);
      warp_m2 = static_cast<T>(0);
      warp_count = static_cast<T>(0);
    }
    __syncwarp();
    T block_mean = 0;
    T block_m2 = 0;
    T block_count = 0;
    WelfordWarpReduce(
        warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_count = count_result_broadcast;
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int kPackSize,
          int block_size>
__global__ void LayerNormBlockSMemImpl(LOAD load,
                                       STORE store,
                                       const int64_t rows,
                                       const int64_t cols,
                                       const double epsilon,
                                       ComputeType* mean,
                                       ComputeType* inv_variance,
                                       ComputeType col_divisor) {
  using LoadType = typename LOAD::LoadType;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<LoadType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % kPackSize == 0);
  const int num_packs = static_cast<int>(cols) / kPackSize;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum = 0;
    ComputeType thread_sum_square = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[kPackSize];
      load.template load<kPackSize>(pack, row, pack_id * kPackSize);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        ComputeType pack_val = static_cast<ComputeType>(pack[i]);
        thread_sum += pack_val;
        thread_sum_square += pack_val * pack_val;
      }
    }

    const ComputeType row_sum =
        BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum);
    const ComputeType row_sum_square =
        BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum_square);

    // use multiply instead of divide.
    ComputeType row_mean = row_sum * col_divisor;
    ComputeType row_sum_square_mean = row_sum_square * col_divisor;
    ComputeType row_variance = max(row_sum_square_mean - row_mean * row_mean,
                                   static_cast<ComputeType>(0.0));
    ComputeType row_inv_var =
        Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0 && mean && inv_variance) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[kPackSize];
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        pack[i] = (static_cast<ComputeType>(buf[i * num_packs + pack_id]) -
                   row_mean) *
                  row_inv_var;
      }
      store.template store<kPackSize>(pack, row, pack_id * kPackSize);
    }
  }
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int pack_size,
          int block_size>
inline cudaError_t LaunchLayerNormBlockSMemImpl(cudaStream_t stream,
                                                LOAD load,
                                                STORE store,
                                                int smem,
                                                const int64_t rows,
                                                const int64_t cols,
                                                const double epsilon,
                                                ComputeType* mean,
                                                ComputeType* inv_variance,
                                                ComputeType col_divisor) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>,
        block_size,
        smem,
        rows,
        waves,
        &grid_dim_x);
    if (err != cudaSuccess) {
      return err;
    }
  }
  LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(
          load, store, rows, cols, epsilon, mean, inv_variance, col_divisor);
  return cudaPeekAtLastError();
}

template <typename Func>
cudaError_t MaximizeDynamicSharedMemorySize(Func func,
                                            const int max_smem_size) {
  cudaFuncAttributes attr{};
  cudaError_t err = cudaFuncGetAttributes(&attr, func);
  if (err != cudaSuccess) {
    return err;
  }
  constexpr int reserved_smem = 1024;  // 1K
  return cudaFuncSetAttribute(
      func,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_smem_size - attr.sharedSizeBytes - reserved_smem);
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t TryDispatchLayerNormBlockSMemImplBlockSize(
    cudaStream_t stream,
    LOAD load,
    STORE store,
    const int64_t rows,
    const int64_t cols,
    const double epsilon,
    ComputeType* mean,
    ComputeType* inv_variance,
    ComputeType col_divisor,
    bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;

  int dev = 0;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }

  const size_t smem = cols * sizeof(typename LOAD::LoadType);

  *success = true;
  return LaunchLayerNormBlockSMemImpl<LOAD,
                                      STORE,
                                      ComputeType,
                                      pack_size,
                                      block_size_conf_1>(stream,
                                                         load,
                                                         store,
                                                         smem,
                                                         rows,
                                                         cols,
                                                         epsilon,
                                                         mean,
                                                         inv_variance,
                                                         col_divisor);
}

template <typename LOAD, typename STORE, typename ComputeType>
struct TryDispatchLayerNormBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream,
                         LOAD load,
                         STORE store,
                         const int64_t rows,
                         const int64_t cols,
                         const double epsilon,
                         ComputeType* mean,
                         ComputeType* inv_variance,
                         ComputeType col_divisor,
                         bool* success) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) &&
        CanPackAs<STORE>(store, 4)) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD,
                                                        STORE,
                                                        ComputeType,
                                                        4>(stream,
                                                           load,
                                                           store,
                                                           rows,
                                                           cols,
                                                           epsilon,
                                                           mean,
                                                           inv_variance,
                                                           col_divisor,
                                                           success);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) &&
               CanPackAs<STORE>(store, 2)) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD,
                                                        STORE,
                                                        ComputeType,
                                                        2>(stream,
                                                           load,
                                                           store,
                                                           rows,
                                                           cols,
                                                           epsilon,
                                                           mean,
                                                           inv_variance,
                                                           col_divisor,
                                                           success);
    } else {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD,
                                                        STORE,
                                                        ComputeType,
                                                        1>(stream,
                                                           load,
                                                           store,
                                                           rows,
                                                           cols,
                                                           epsilon,
                                                           mean,
                                                           inv_variance,
                                                           col_divisor,
                                                           success);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t TryDispatchLayerNormBlockSMemImpl(cudaStream_t stream,
                                                     LOAD load,
                                                     STORE store,
                                                     const int64_t rows,
                                                     const int64_t cols,
                                                     const double epsilon,
                                                     ComputeType* mean,
                                                     ComputeType* inv_variance,
                                                     ComputeType col_divisor,
                                                     bool* success) {
  return TryDispatchLayerNormBlockSMemImplPackSize<LOAD, STORE, ComputeType>()(
      stream,
      load,
      store,
      rows,
      cols,
      epsilon,
      mean,
      inv_variance,
      col_divisor,
      success);
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int kPackSize,
          int block_size>
__global__ void __launch_bounds__(1024)
    LayerNormBlockUncachedImpl(LOAD load,
                               STORE store,
                               const int64_t rows,
                               const int64_t cols,
                               const double epsilon,
                               ComputeType* mean,
                               ComputeType* inv_variance) {
  using LoadType = typename LOAD::LoadType;
  const int tid = threadIdx.x;
  assert(cols % kPackSize == 0);
  const int num_packs = static_cast<int>(cols) / kPackSize;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[kPackSize];
      load.template load<kPackSize>(pack, row, pack_id * kPackSize);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        WelfordCombine(static_cast<ComputeType>(pack[i]),
                       &thread_mean,
                       &thread_m2,
                       &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType>(
        thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);
    ComputeType row_variance =
        max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var =
        Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0 && mean && inv_variance) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[kPackSize];
      ComputeType dst_pack[kPackSize];
      const int pack_offset = pack_id * kPackSize;
      load.template load<kPackSize>(pack, row, pack_offset);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        dst_pack[i] =
            (static_cast<ComputeType>(pack[i]) - row_mean) * row_inv_var;
      }
      store.template store<kPackSize>(dst_pack, row, pack_offset);
    }
  }
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline cudaError_t LaunchLayerNormBlockUncachedImpl(cudaStream_t stream,
                                                    LOAD load,
                                                    STORE store,
                                                    const int64_t rows,
                                                    const int64_t cols,
                                                    const double epsilon,
                                                    ComputeType* mean,
                                                    ComputeType* inv_variance) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(LayerNormBlockUncachedImpl<LOAD,
                                                              STORE,
                                                              ComputeType,
                                                              pack_size,
                                                              block_size>,
                                   block_size,
                                   0,
                                   rows,
                                   waves,
                                   &grid_dim_x);
    if (err != cudaSuccess) {
      return err;
    }
  }
  LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(
          load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template <typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream,
                         LOAD load,
                         STORE store,
                         const int64_t rows,
                         const int64_t cols,
                         const double epsilon,
                         ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) &&
        CanPackAs<STORE>(store, 4)) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 4>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) &&
               CanPackAs<STORE>(store, 2)) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 2>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 1>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t DispatchLayerNormBlockUncachedImpl(
    cudaStream_t stream,
    LOAD load,
    STORE store,
    const int64_t rows,
    const int64_t cols,
    const double epsilon,
    ComputeType* mean,
    ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value,
                               cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream,
                  LOAD load,
                  STORE store,
                  const int64_t rows,
                  const int64_t cols,
                  const double epsilon,
                  ComputeType* mean,
                  ComputeType* inv_variance) {
  const ComputeType col_divisor = 1.0f / cols;
  bool dispatch_smem_impl_success;
  {
    cudaError_t err =
        TryDispatchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType>(
            stream,
            load,
            store,
            rows,
            cols,
            epsilon,
            mean,
            inv_variance,
            col_divisor,
            &dispatch_smem_impl_success);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (!dispatch_smem_impl_success) {
    return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
  return cudaSuccess;
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value,
                               cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream,
                  LOAD load,
                  STORE store,
                  const int64_t rows,
                  const int64_t cols,
                  const double epsilon,
                  ComputeType* mean,
                  ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template <typename InputType, typename SRC, typename DST>
struct DequantSkipLoad {
  using LoadType = DST;
  DequantSkipLoad(const InputType* src,
                  const SRC* bias,
                  const SRC* skip,
                  const float* dequant_scale,
                  float alpha,
                  int64_t row_size)
      : src(src),
        bias(bias),
        skip(skip),
        dequant_scale(dequant_scale),
        alpha(alpha),
        row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<InputType, N> src_pack;
    Pack<SRC, N> bias_pack;
    Pack<SRC, N> skip_pack;
    Pack<float, N> dequant_scale_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t bias_offset = col / N;
    src_pack = *(reinterpret_cast<const Pack<InputType, N>*>(src) + offset);
    bias_pack = *(reinterpret_cast<const Pack<SRC, N>*>(bias) + bias_offset);
    skip_pack = *(reinterpret_cast<const Pack<SRC, N>*>(skip) + offset);
    dequant_scale_pack =
        *(reinterpret_cast<const Pack<float, N>*>(dequant_scale) +
          bias_offset);  // equal to col.
#pragma unroll
    for (int i = 0; i < N; ++i) {
      // First we need to cast src and dequant.
      dst[i] = static_cast<DST>(
          static_cast<DST>(static_cast<float>(src_pack.elem[i]) *
                           dequant_scale_pack.elem[i]) +
          bias_pack.elem[i] + skip_pack.elem[i]);
    }
  }
  const InputType* src;
  const SRC* bias;
  const SRC* skip;
  const float* dequant_scale;
  double alpha;
  int64_t row_size;
};

template <typename T>
__device__ __inline__ T ClipFunc(const T v, const T min, const T max) {
  if (v > max) return max;
  if (v < min) return min;
  return v;
}

template <typename InType, typename OutType>
__forceinline__ __device__ OutType QuantHelperFunc(const InType input,
                                                   const float scale,
                                                   const int round_type,
                                                   const float max_bound,
                                                   const float min_bound) {
  float quant_value = max_bound * scale * input;

  if (round_type == 0) {
    quant_value = static_cast<float>(rint(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <typename OutType,
          typename SRC,
          typename DST,
          bool do_scale,
          bool do_center>
struct AffineQuantStore {
  AffineQuantStore(OutType* y,
                   const int64_t row_size,
                   const float* gamma,
                   const float* beta,
                   const float quant_out_scale,
                   const int quant_round_type = 1,
                   const float quant_max_bound = 127.0,
                   const float quant_min_bound = -127.0)
      : y(y),
        row_size(row_size),
        gamma(gamma),
        beta(beta),
        quant_round_type(quant_round_type),
        quant_out_scale(quant_out_scale),
        quant_max_bound(quant_max_bound),
        quant_min_bound(quant_min_bound) {}

  template <int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<OutType, N> y_pack;
    Pack<float, N> gamma_pack;
    Pack<float, N> beta_pack;
    Pack<OutType, N> out_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    gamma_pack =
        *(reinterpret_cast<const Pack<float, N>*>(gamma) + gamma_offset);
    beta_pack = *(reinterpret_cast<const Pack<float, N>*>(beta) + gamma_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float normalized_i = static_cast<float>(src[i]);
      float normalized_val =
          normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      y_pack.elem[i] = QuantHelperFunc<float, OutType>(normalized_val,
                                                       quant_out_scale,
                                                       quant_round_type,
                                                       quant_max_bound,
                                                       quant_min_bound);
    }
    *(reinterpret_cast<Pack<OutType, N>*>(y) + offset) = y_pack;
  }

  OutType* y;
  int64_t row_size;
  const float* gamma;
  const float* beta;
  const int quant_round_type;
  const float quant_out_scale;
  const float quant_max_bound;
  const float quant_min_bound;
};

template <typename T, typename SRC, bool do_scale, bool do_center>
struct AffineStore {
  AffineStore(T* y,
              const int64_t row_size,
              const float* gamma,
              const float* beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}

  template <int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<T, N> y_pack;
    Pack<float, N> gamma_pack;
    Pack<float, N> beta_pack;
    Pack<T, N> out_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    gamma_pack =
        *(reinterpret_cast<const Pack<float, N>*>(gamma) + gamma_offset);
    beta_pack = *(reinterpret_cast<const Pack<float, N>*>(beta) + gamma_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float normalized_i = static_cast<float>(src[i]);
      float normalized_val =
          normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      y_pack.elem[i] = static_cast<T>(normalized_val);
    }
    *(reinterpret_cast<Pack<T, N>*>(y) + offset) = y_pack;
  }

  T* y;
  int64_t row_size;
  const float* gamma;
  const float* beta;
};

template <typename InputType, typename SRC, typename DST>
struct DequantSkipLoadAndStoreResidual {
  using LoadType = DST;
  // need to aseert SRC equals to DST.
  DequantSkipLoadAndStoreResidual(const InputType* src,
                                  const SRC* bias,
                                  const SRC* skip,
                                  const float* dequant_scale,
                                  SRC* residual_bias_out,
                                  float alpha,
                                  int64_t row_size)
      : src(src),
        bias(bias),
        skip(skip),
        dequant_scale(dequant_scale),
        residual_bias_out(residual_bias_out),
        alpha(alpha),
        row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<InputType, N> src_pack;
    Pack<SRC, N> bias_pack;
    Pack<SRC, N> skip_pack;
    Pack<float, N> dequant_scale_pack;
    Pack<DST, N> residual_out_pack;

    const int64_t offset = (row * row_size + col) / N;
    const int64_t bias_offset = col / N;
    src_pack = *(reinterpret_cast<const Pack<InputType, N>*>(src) + offset);
    bias_pack = *(reinterpret_cast<const Pack<SRC, N>*>(bias) + bias_offset);
    skip_pack = *(reinterpret_cast<const Pack<SRC, N>*>(skip) + offset);
    dequant_scale_pack =
        *(reinterpret_cast<const Pack<float, N>*>(dequant_scale) +
          bias_offset);  // equal to col.
#pragma unroll
    for (int i = 0; i < N; ++i) {
      // First we need to cast src and dequant.
      residual_out_pack.elem[i] = static_cast<DST>(
          static_cast<DST>(static_cast<float>(src_pack.elem[i]) *
                           dequant_scale_pack.elem[i]) +
          bias_pack.elem[i] + skip_pack.elem[i]);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = residual_out_pack.elem[i];
    }
    *(reinterpret_cast<Pack<SRC, N>*>(residual_bias_out) + offset) =
        residual_out_pack;
  }
  const InputType* src;
  const SRC* bias;
  const SRC* skip;
  const float* dequant_scale;
  SRC* residual_bias_out;
  double alpha;
  int64_t row_size;
};

template <typename T>
struct SkipLoadAndStoreResidual {
  using LoadType = T;
  SkipLoadAndStoreResidual(const T* src,
                           const T* bias,
                           const T* skip,
                           T* residual_bias_out,
                           float alpha,
                           int64_t row_size)
      : src(src),
        bias(bias),
        skip(skip),
        residual_bias_out(residual_bias_out),
        alpha(alpha),
        row_size(row_size) {}
  template <int N>
  __device__ void load(T* dst, int64_t row, int64_t col) const {
    Pack<T, N> src_pack;
    Pack<T, N> bias_pack;
    Pack<T, N> skip_pack;
    Pack<T, N> residual_out_pack;

    const int64_t offset = (row * row_size + col) / N;
    const int64_t bias_offset = col / N;
    src_pack = *(reinterpret_cast<const Pack<T, N>*>(src) + offset);
    bias_pack = *(reinterpret_cast<const Pack<T, N>*>(bias) + bias_offset);
    skip_pack = *(reinterpret_cast<const Pack<T, N>*>(skip) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      // First we need to cast src and dequant.
      residual_out_pack.elem[i] =
          static_cast<T>(static_cast<T>(static_cast<float>(src_pack.elem[i])) +
                         bias_pack.elem[i] + skip_pack.elem[i]);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = residual_out_pack.elem[i];
    }
    *(reinterpret_cast<Pack<T, N>*>(residual_bias_out) + offset) =
        residual_out_pack;
  }
  const T* src;
  const T* bias;
  const T* skip;
  T* residual_bias_out;
  double alpha;
  int64_t row_size;
};

#endif

}  // namespace

// ===== FP16 in, Int8out LayerNormQuantDequant =====

template <typename T, typename Context>
void ResidualAddLayerNormQuantDequantKernel(const Context& dev_ctx,
                                            const DenseTensor& x,
                                            const DenseTensor& residual,
                                            const DenseTensor& bias,
                                            const DenseTensor& norm_weight,
                                            const DenseTensor& norm_bias,
                                            float epsilon,
                                            const float in_scale,
                                            const int quant_round_type,
                                            const float quant_max_bound,
                                            const float quant_min_bound,
                                            int begin_norm_axis,
                                            DenseTensor* residual_out,
                                            DenseTensor* out) {
#if defined(PADDLE_WITH_HIP)
  LOG(ERROR) << "Please compile with CUDA, ROCM platform isn't support it";
#else
  using U = phi::funcs::LayerNormParamType<T>;
  const T* x_data = x.data<T>();
  const T* residual_data = residual.data<T>();
  const T* bias_data = bias.data<T>();
  const float* norm_weight_data = norm_weight.data<float>();
  const float* norm_bias_data = norm_bias.data<float>();
  int8_t* out_data = dev_ctx.template Alloc<int8_t>(out);
  T* residual_out_data = dev_ctx.template Alloc<T>(residual_out);

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  SkipLoadAndStoreResidual<T> load(
      x_data, bias_data, residual_data, residual_out_data, 0.0f, cols);

  AffineQuantStore<int8_t, U, T, true, true> store(out,
                                                   cols,
                                                   norm_weight_data,
                                                   norm_bias_data,
                                                   in_scale,
                                                   quant_round_type,
                                                   quant_max_bound,
                                                   quant_min_bound);
  DispatchLayerNorm<decltype(load), decltype(store), U>(
      dev_ctx.stream(),
      load,
      store,
      rows,
      cols,
      epsilon,
      nullptr /*ln_mean_data*/,
      nullptr /*ln_var_data*/);
#endif
}

template <typename T, typename Context>
void ResidualAddLayerNormQuantDequantWrapper(const Context& ctx,
                                             const T* x,
                                             const T* residual,
                                             const T* bias,
                                             const float* norm_weight,
                                             const float* norm_bias,
                                             const float epsilon,
                                             const int rows,
                                             const int cols,
                                             const float in_scale,
                                             const int quant_round_type,
                                             const float quant_max_bound,
                                             const float quant_min_bound,
                                             T* residual_out,
                                             int8_t* out) {
#if defined(PADDLE_WITH_HIP)
  LOG(ERROR) << "Please compile with CUDA, ROCM platform isn't support it";
#else
  using U = phi::funcs::LayerNormParamType<T>;
  SkipLoadAndStoreResidual<T> load(x, bias, residual, residual_out, 0.0f, cols);

  AffineQuantStore<int8_t, U, T, true, true> store(out,
                                                   cols,
                                                   norm_weight,
                                                   norm_bias,
                                                   in_scale,
                                                   quant_round_type,
                                                   quant_max_bound,
                                                   quant_min_bound);
  DispatchLayerNorm<decltype(load), decltype(store), U>(
      ctx.stream(),
      load,
      store,
      rows,
      cols,
      epsilon,
      nullptr /*ln_mean_data*/,
      nullptr /*ln_var_data*/);
#endif
}

template void ResidualAddLayerNormQuantDequantWrapper(
    const phi::GPUContext& ctx,
    const float* x,
    const float* residual,
    const float* bias,
    const float* norm_weight,
    const float* norm_bias,
    const float epsilon,
    const int rows,
    const int cols,
    const float in_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    float* residual_out,
    int8_t* out);

template void ResidualAddLayerNormQuantDequantWrapper(
    const phi::GPUContext& ctx,
    const phi::dtype::float16* x,
    const phi::dtype::float16* residual,
    const phi::dtype::float16* bias,
    const float* norm_weight,
    const float* norm_bias,
    const float epsilon,
    const int rows,
    const int cols,
    const float in_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    phi::dtype::float16* residual_out,
    int8_t* out);

template void ResidualAddLayerNormQuantDequantWrapper(
    const phi::GPUContext& ctx,
    const phi::dtype::bfloat16* x,
    const phi::dtype::bfloat16* residual,
    const phi::dtype::bfloat16* bias,
    const float* norm_weight,
    const float* norm_bias,
    const float epsilon,
    const int rows,
    const int cols,
    const float in_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    phi::dtype::bfloat16* residual_out,
    int8_t* out);

}  // namespace phi

PD_REGISTER_KERNEL(layer_norm_int8,
                   GPU,
                   ALL_LAYOUT,
                   phi::ResidualAddLayerNormQuantDequantKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
