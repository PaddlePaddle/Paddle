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
// single Pass algorithm. Support Int8 quant, dequant Load/Store implementation.

#include <assert.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/attention_layer.norm.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_helper.h"
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#define GPU(str) hip##str
#define GPUMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#else
#include <cub/cub.cuh>
#define GPU(str) cuda##str
#define GPUMultiProcessorCount cudaDevAttrMultiProcessorCount
#endif

namespace phi {

namespace fusion {

namespace {

#ifdef PADDLE_WITH_HIP
constexpr int kWarpSize = 64;
#else
constexpr int kWarpSize = 32;
#endif

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
#ifdef PADDLE_WITH_HIP
        val, __shfl_xor(val, mask, thread_group_width));
#else
        val, __shfl_xor_sync(0xffffffff, val, mask, thread_group_width));
#endif
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
#if defined(OF_LAYER_NORM_USE_FAST_MATH) || defined(PADDLE_WITH_HIP)
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
#if defined(OF_LAYER_NORM_USE_FAST_MATH) || defined(PADDLE_WITH_HIP)
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
inline GPU(Error_t) GetNumBlocks(Func func,
                                 int64_t block_size,
                                 size_t dynamic_smem_size,
                                 int64_t max_blocks,
                                 int64_t waves,
                                 int* num_blocks) {
  int dev;
  {
    GPU(Error_t) err = GPU(GetDevice)(&dev);
    if (err != GPU(Success)) {
      return err;
    }
  }
  int sm_count;
  {
    GPU(Error_t)
    err = GPU(DeviceGetAttribute)(&sm_count, GPUMultiProcessorCount, dev);
    if (err != GPU(Success)) {
      return err;
    }
  }
  int max_active_blocks;
  {
    GPU(Error_t)
    err = GPU(OccupancyMaxActiveBlocksPerMultiprocessor)(
        &max_active_blocks, func, block_size, dynamic_smem_size);
  }
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return GPU(Success);
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
  // Use Welford Online algorithm to compute mean and variance
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
#ifdef PADDLE_WITH_HIP
    T b_mean = __shfl_down(*mean, mask, thread_group_width);
    T b_m2 = __shfl_down(*m2, mask, thread_group_width);
    T b_count = __shfl_down(*count, mask, thread_group_width);
#else
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
#endif
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(
    T thread_mean, T thread_m2, T thread_count, T* mean, T* m2, T* count) {
  WelfordWarpReduce<T, thread_group_width>(
      thread_mean, thread_m2, thread_count, mean, m2, count);
#ifdef PADDLE_WITH_HIP
  *mean = __shfl(*mean, 0, thread_group_width);
  *m2 = __shfl(*m2, 0, thread_group_width);
  *count = __shfl(*count, 0, thread_group_width);
#else
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
#endif
}

template <typename T, int thread_group_width = kWarpSize>
__inline__ __device__ T WarpReduceSum(T x) {
  T result = 0.0f;
#pragma unroll
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
#ifdef PADDLE_WITH_HIP
    result += __shfl_xor(x, mask, thread_group_width);
#else
    result += __shfl_xor_sync(0xffffffff, x, mask, thread_group_width);
#endif
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
#ifdef PADDLE_WITH_HIP
    __syncthreads();
#else
    __syncwarp();
#endif
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
inline GPU(Error_t) LaunchLayerNormBlockSMemImpl(GPU(Stream_t) stream,
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
    GPU(Error_t)
    err = GetNumBlocks(
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>,
        block_size,
        smem,
        rows,
        waves,
        &grid_dim_x);
    if (err != GPU(Success)) {
      return err;
    }
  }
  LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(
          load, store, rows, cols, epsilon, mean, inv_variance, col_divisor);
  return GPU(PeekAtLastError)();
}

template <typename Func>
GPU(Error_t)
MaximizeDynamicSharedMemorySize(Func func, const int max_smem_size) {
  GPU(FuncAttributes) attr{};
#ifdef PADDLE_WITH_HIP
  hipError_t err = hipFuncGetAttributes(&attr, (const void*)func);
#else
  cudaError_t err = cudaFuncGetAttributes(&attr, func);
#endif
  if (err != GPU(Success)) {
    return err;
  }
  constexpr int reserved_smem = 1024;  // 1K
#ifdef PADDLE_WITH_HIP
  return hipFuncSetAttribute(
      (const void*)func,
      hipFuncAttributeMaxDynamicSharedMemorySize,
      max_smem_size - attr.sharedSizeBytes - reserved_smem);
#else
  return cudaFuncSetAttribute(
      func,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_smem_size - attr.sharedSizeBytes - reserved_smem);
#endif
}

template <typename LOAD, typename STORE, typename ComputeType, int pack_size>
inline GPU(Error_t)
    TryDispatchLayerNormBlockSMemImplBlockSize(GPU(Stream_t) stream,
                                               LOAD load,
                                               STORE store,
                                               const int64_t rows,
                                               const int64_t cols,
                                               const double epsilon,
                                               ComputeType* mean,
                                               ComputeType* inv_variance,
                                               ComputeType col_divisor,
                                               bool* success) {
  // Note(Zhengzekang): We choose a fixed blocksize to avoid layernorm diff, by
  // RichardWooSJTU.

  constexpr int block_size_conf_1 = 512;

  int dev = 0;
  {
    GPU(Error_t) err = GPU(GetDevice)(&dev);
    if (err != GPU(Success)) {
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
  GPU(Error_t)
  operator()(GPU(Stream_t) stream,
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
inline GPU(Error_t) TryDispatchLayerNormBlockSMemImpl(GPU(Stream_t) stream,
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
inline GPU(Error_t)
    LaunchLayerNormBlockUncachedImpl(GPU(Stream_t) stream,
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
    GPU(Error_t)
    err = GetNumBlocks(LayerNormBlockUncachedImpl<LOAD,
                                                  STORE,
                                                  ComputeType,
                                                  pack_size,
                                                  block_size>,
                       block_size,
                       0,
                       rows,
                       waves,
                       &grid_dim_x);
    if (err != GPU(Success)) {
      return err;
    }
  }
  LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(
          load, store, rows, cols, epsilon, mean, inv_variance);
  return GPU(PeekAtLastError)();
}

template <typename LOAD, typename STORE, typename ComputeType>
struct DispatchLayerNormBlockUncachedImplPackSize {
  GPU(Error_t)
  operator()(GPU(Stream_t) stream,
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
inline GPU(Error_t)
    DispatchLayerNormBlockUncachedImpl(GPU(Stream_t) stream,
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
                               GPU(Error_t)>::type
DispatchLayerNorm(GPU(Stream_t) stream,
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
    GPU(Error_t)
    err = TryDispatchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType>(
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
    if (err != GPU(Success)) {
      return err;
    }
  }
  if (!dispatch_smem_impl_success) {
    return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
  return GPU(Success);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value,
                               GPU(Error_t)>::type
DispatchLayerNorm(GPU(Stream_t) stream,
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

template <typename InType, typename OutType>
__forceinline__ __device__ OutType FP8QuantHelperFunc(const InType input,
                                                      const float scale,
                                                      const int round_type,
                                                      const float max_bound,
                                                      const float min_bound) {
  float quant_value = max_bound * scale * input;
  // float quant_value = input;
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
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound)
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
      if constexpr (std::is_same_v<OutType, phi::dtype::float8_e4m3fn>) {
        y_pack.elem[i] = FP8QuantHelperFunc<float, OutType>(normalized_val,
                                                            quant_out_scale,
                                                            quant_round_type,
                                                            quant_max_bound,
                                                            quant_min_bound);
      } else {
        y_pack.elem[i] = QuantHelperFunc<float, OutType>(normalized_val,
                                                         quant_out_scale,
                                                         quant_round_type,
                                                         quant_max_bound,
                                                         quant_min_bound);
      }
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
    T alpha_val = static_cast<T>(alpha);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      // First we need to cast src and dequant.
      residual_out_pack.elem[i] =
          static_cast<T>(static_cast<T>(static_cast<float>(src_pack.elem[i])) +
                         bias_pack.elem[i] + skip_pack.elem[i] * alpha_val);
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
  float alpha;
  int64_t row_size;
};

}  // namespace

template <typename T, typename Context>
void FusedLayerNormKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const paddle::optional<DenseTensor>& bias,
                          const paddle::optional<DenseTensor>& residual,
                          const paddle::optional<DenseTensor>& norm_weight,
                          const paddle::optional<DenseTensor>& norm_bias,
                          const float epsilon,
                          const float residual_alpha,
                          const int begin_norm_axis,
                          const float quant_scale,
                          const int quant_round_type,
                          const float quant_max_bound,
                          const float quant_min_bound,
                          DenseTensor* out,
                          DenseTensor* residual_out,
                          DenseTensor* mean,
                          DenseTensor* variance) {
  if (out->dtype() == phi::DataType::INT8 ||
      out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
    PADDLE_ENFORCE_EQ(
        quant_scale != 0.0f,
        true,
        common::errors::InvalidArgument(
            "Quant fused_bias_residual_layernorm'output, must has quant_scale, "
            "quant_scale!=0, but quant_scale = %f ",
            quant_scale));
    PADDLE_ENFORCE_EQ(quant_round_type == 0 || quant_round_type == 1,
                      true,
                      common::errors::InvalidArgument(
                          "Quant fused_bias_residual_layernorm'output, must "
                          "has quant_round_type, "
                          "quant_round_type = 0 or quant_round_type = 1, but "
                          "quant_scale = %d ",
                          quant_scale));
    PADDLE_ENFORCE_EQ(quant_max_bound != 0.0f,
                      true,
                      common::errors::InvalidArgument(
                          "Quant fused_bias_residual_layernorm'output, must "
                          "has quant_max_bound and "
                          "quant_max_bound!=0, but quant_max_bound = %f ",
                          quant_scale));
    PADDLE_ENFORCE_EQ(quant_min_bound != 0.0f,
                      true,
                      common::errors::InvalidArgument(
                          "Quant fused_bias_residual_layernorm'output, must "
                          "has quant_min_bound and "
                          "quant_min_bound!=0, but quant_min_bound = %f ",
                          quant_scale));
  }

  using U = phi::funcs::LayerNormParamType<T>;
  const T* x_data = x.data<T>();
  const U* norm_weight_data =
      norm_weight ? norm_weight.get().data<U>() : nullptr;
  const U* norm_bias_data = norm_bias ? norm_bias.get().data<U>() : nullptr;
  int64_t rows = 1;
  int64_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  phi::fusion::DropoutParam dropout_param(true, 0, true, true, 0.0, nullptr, 0);
  phi::fusion::FusedDropoutLayerNormHelper<T, uint8_t>
      residual_bias_add_layernorm_helper(
          dev_ctx, rows, cols, dropout_param, epsilon, residual_alpha);
  phi::fusion::AttnLayerNorm<T> layernorm_helper(dev_ctx, epsilon, rows, cols);

  // Do residual + bias + x
  if (residual && norm_weight_data == nullptr && norm_bias_data == nullptr) {
    const T* residual_data = residual.get().data<T>();
    const T* bias_data = bias ? bias.get().data<T>() : nullptr;
    T* out_data = dev_ctx.template Alloc<T>(out);
    residual_bias_add_layernorm_helper.ResidualDropoutBias(
        dev_ctx,
        x_data,
        residual_data,
        bias_data,
        out_data,
        nullptr /*dropout_mask_out_data*/);
    return;
  }

  U* mean_data = dev_ctx.template Alloc<U>(mean);
  U* variance_data = dev_ctx.template Alloc<U>(variance);

  if (residual) {
    // Do Layernorm(residual + bias + x)
    T* residual_out_data = dev_ctx.template Alloc<T>(residual_out);
    const T* residual_data = residual.get().data<T>();
    const T* bias_data = bias ? bias.get().data<T>() : nullptr;
    if (out->dtype() == phi::DataType::INT8) {
      // Quantize and output int8.
      int8_t* out_data = dev_ctx.template Alloc<int8_t>(out);
      SkipLoadAndStoreResidual<T> load(x_data,
                                       bias_data,
                                       residual_data,
                                       residual_out_data,
                                       residual_alpha,
                                       cols);
      AffineQuantStore<int8_t, U, T, true, true> store(out_data,
                                                       cols,
                                                       norm_weight_data,
                                                       norm_bias_data,
                                                       quant_scale,
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
          mean_data /*ln_mean_data*/,
          variance_data /*ln_var_data*/);
    } else if (out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
      // Quantize and output float8_e4m3fn.
      phi::dtype::float8_e4m3fn* out_data =
          dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
      SkipLoadAndStoreResidual<T> load(x_data,
                                       bias_data,
                                       residual_data,
                                       residual_out_data,
                                       residual_alpha,
                                       cols);
      AffineQuantStore<phi::dtype::float8_e4m3fn, U, T, true, true> store(
          out_data,
          cols,
          norm_weight_data,
          norm_bias_data,
          quant_scale,
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
          mean_data /*ln_mean_data*/,
          variance_data /*ln_var_data*/);
    } else {
      // No Quantize.
      T* out_data = dev_ctx.template Alloc<T>(out);
      residual_bias_add_layernorm_helper.LayernormResidualDropoutBias(
          dev_ctx,
          x_data,
          residual_data,
          bias_data,
          norm_weight_data,
          norm_bias_data,
          residual_out_data,
          nullptr,
          out_data,
          mean_data,
          variance_data);
    }
  } else {
    if (out->dtype() == phi::DataType::INT8) {
      // Quantize and output int8.
      int8_t* out_data = dev_ctx.template Alloc<int8_t>(out);
      DirectLoad<T, U> load(x_data, cols);
      AffineQuantStore<int8_t, U, T, true, true> store(out_data,
                                                       cols,
                                                       norm_weight_data,
                                                       norm_bias_data,
                                                       quant_scale,
                                                       quant_round_type,
                                                       quant_max_bound,
                                                       quant_min_bound);
      DispatchLayerNorm<decltype(load), decltype(store), U>(dev_ctx.stream(),
                                                            load,
                                                            store,
                                                            rows,
                                                            cols,
                                                            epsilon,
                                                            mean_data,
                                                            variance_data);
    } else if (out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
      // Quantize and output float8_e4m3fn.
      phi::dtype::float8_e4m3fn* out_data =
          dev_ctx.template Alloc<phi::dtype::float8_e4m3fn>(out);
      DirectLoad<T, U> load(x_data, cols);
      AffineQuantStore<phi::dtype::float8_e4m3fn, U, T, true, true> store(
          out_data,
          cols,
          norm_weight_data,
          norm_bias_data,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
      DispatchLayerNorm<decltype(load), decltype(store), U>(dev_ctx.stream(),
                                                            load,
                                                            store,
                                                            rows,
                                                            cols,
                                                            epsilon,
                                                            mean_data,
                                                            variance_data);
    } else {
      // No Quantize.
      T* out_data = dev_ctx.template Alloc<T>(out);
      layernorm_helper.ComputeForward(x_data,
                                      norm_weight_data,
                                      norm_bias_data,
                                      out_data,
                                      mean_data,
                                      variance_data);
    }
  }
}

}  // namespace fusion

}  // namespace phi

#ifndef PADDLE_WITH_HIP
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(fused_bias_residual_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLayerNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
#else
PD_REGISTER_KERNEL(fused_bias_residual_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLayerNormKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
#endif  // CUDNN_VERSION_MIN
#else
PD_REGISTER_KERNEL(fused_bias_residual_layernorm,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLayerNormKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
}
#endif  // PADDLE_WITH_HIP
