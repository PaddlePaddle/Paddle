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

#include "paddle/phi/kernels/rms_norm_kernel.h"
#include <assert.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#ifndef PADDLE_WITH_HIP
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
  return a / b;
}

template <>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

template <typename T>
__inline__ __device__ T Rsqrt(T x);

template <>
__inline__ __device__ float Rsqrt<float>(float x) {
  return rsqrt(x);
}

template <>
__inline__ __device__ double Rsqrt<double>(double x) {
  return rsqrt(x);
}

template <class Func>
inline cudaError_t GetNumBlocks(Func func,
                                int32_t block_size,
                                size_t dynamic_smem_size,
                                int32_t max_blocks,
                                int32_t waves,
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
      1, std::min<int32_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

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
  __device__ Pack() = default;
  T elem[N];
};

template <typename SRC, typename DST>
struct DirectLoad {
  using LoadType = DST;
  DirectLoad(const SRC* src, int32_t row_size) : src(src), row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    Pack<SRC, N> pack;
    const int32_t offset = (row * row_size + col) / N;
    pack = *(reinterpret_cast<const Pack<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]);
    }
  }
  const SRC* src;
  int32_t row_size;
};

template <typename SRC, typename DST>
struct ResidualAddBiasLoad {
  using LoadType = DST;
  ResidualAddBiasLoad(const SRC* src,
                      const SRC* residual,
                      const SRC* bias,
                      SRC* residual_out,
                      int32_t row_size)
      : src(src),
        residual(residual),
        bias(bias),
        residual_out(residual_out),
        row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    Pack<SRC, N> src_pack;
    Pack<SRC, N> residual_pack;
    Pack<SRC, N> bias_pack;
    const int32_t offset = (row * row_size + col) / N;

    src_pack = *(reinterpret_cast<const Pack<SRC, N>*>(src) + offset);
    residual_pack = *(reinterpret_cast<const Pack<SRC, N>*>(residual) + offset);

    if (bias) {
      bias_pack = *(reinterpret_cast<const Pack<SRC, N>*>(bias) + col / N);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        bias_pack.elem[i] = static_cast<SRC>(0.0f);
      }
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      src_pack.elem[i] =
          src_pack.elem[i] + residual_pack.elem[i] + bias_pack.elem[i];
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(src_pack.elem[i]);
    }
    *(reinterpret_cast<Pack<SRC, N>*>(residual_out) + offset) = src_pack;
  }
  const SRC* src;
  const SRC* residual;
  const SRC* bias;
  SRC* residual_out;
  int32_t row_size;
};

template <typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int32_t row_size) : dst(dst), row_size(row_size) {}
  template <int N>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    Pack<DST, N> pack;
    const int32_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      pack.elem[i] = static_cast<DST>(src[i]);
    }
    *(reinterpret_cast<Pack<DST, N>*>(dst) + offset) = pack;
  }
  DST* dst;
  int32_t row_size;
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
__global__ void __launch_bounds__(block_size)
    RmsNormBlockSMemImpl(LOAD load,
                         STORE store,
                         const int32_t rows,
                         const int32_t cols,
                         const float epsilon,
                         ComputeType col_divisor) {
  using LoadType = typename LOAD::LoadType;
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<LoadType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % kPackSize == 0);
  const int num_packs = static_cast<int>(cols) / kPackSize;
  for (int32_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_sum_square = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      LoadType pack[kPackSize];
      load.template load<kPackSize>(pack, row, pack_id * kPackSize);
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        ComputeType pack_val = static_cast<ComputeType>(pack[i]);
        thread_sum_square += pack_val * pack_val;
      }
    }

    const ComputeType row_sum_square =
        BlockAllReduce<SumOp, ComputeType, block_size>(thread_sum_square);

    // use multiply instead of divide. Author(zhengzekang).
    ComputeType row_rms = row_sum_square * col_divisor;
    ComputeType row_inv_rms =
        Rsqrt(row_rms + static_cast<ComputeType>(epsilon));
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[kPackSize];
#pragma unroll
      for (int i = 0; i < kPackSize; ++i) {
        pack[i] = static_cast<ComputeType>(buf[i * num_packs + pack_id]) *
                  row_inv_rms;
      }
      store.template store<kPackSize>(pack, row, pack_id * kPackSize);
    }
  }
}

template <typename LOAD,
          typename STORE,
          typename ComputeType,
          int kPackSize,
          int block_size>
inline cudaError_t LaunchRmsNormBlockSMemImpl(cudaStream_t stream,
                                              LOAD load,
                                              STORE store,
                                              int smem,
                                              const int32_t rows,
                                              const int32_t cols,
                                              const float epsilon,
                                              ComputeType col_divisor) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(
        RmsNormBlockSMemImpl<LOAD, STORE, ComputeType, kPackSize, block_size>,
        block_size,
        smem,
        rows,
        waves,
        &grid_dim_x);
    if (err != cudaSuccess) {
      return err;
    }
  }
  RmsNormBlockSMemImpl<LOAD, STORE, ComputeType, kPackSize, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(
          load, store, rows, cols, epsilon, col_divisor);
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

template <typename LOAD, typename STORE, typename ComputeType, int kPackSize>
inline cudaError_t TryDispatchRmsNormBlockSMemImplBlockSize(
    cudaStream_t stream,
    LOAD load,
    STORE store,
    const int32_t rows,
    const int32_t cols,
    const float epsilon,
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

  int sm_count = 0;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }

  static const bool max_smem_configed = [=]() {
    int max_smem_size = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (err != cudaSuccess) {
      return false;
    }

    err =
        MaximizeDynamicSharedMemorySize(RmsNormBlockSMemImpl<LOAD,
                                                             STORE,
                                                             ComputeType,
                                                             kPackSize,
                                                             block_size_conf_1>,
                                        max_smem_size);
    if (err != cudaSuccess) {
      return false;
    }
    err =
        MaximizeDynamicSharedMemorySize(RmsNormBlockSMemImpl<LOAD,
                                                             STORE,
                                                             ComputeType,
                                                             kPackSize,
                                                             block_size_conf_2>,
                                        max_smem_size);
    if (err != cudaSuccess) {
      return false;
    }
    err =
        MaximizeDynamicSharedMemorySize(RmsNormBlockSMemImpl<LOAD,
                                                             STORE,
                                                             ComputeType,
                                                             kPackSize,
                                                             block_size_conf_3>,
                                        max_smem_size);
    if (err != cudaSuccess) {
      return false;
    }
    err =
        MaximizeDynamicSharedMemorySize(RmsNormBlockSMemImpl<LOAD,
                                                             STORE,
                                                             ComputeType,
                                                             kPackSize,
                                                             block_size_conf_4>,
                                        max_smem_size);
    if (err != cudaSuccess) {
      return false;
    }

    return true;
  }();

  const size_t smem = cols * sizeof(typename LOAD::LoadType);

  int max_active_blocks_conf_1;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        RmsNormBlockSMemImpl<LOAD,
                             STORE,
                             ComputeType,
                             kPackSize,
                             block_size_conf_1>,
        block_size_conf_1,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }

  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        RmsNormBlockSMemImpl<LOAD,
                             STORE,
                             ComputeType,
                             kPackSize,
                             block_size_conf_4>,
        block_size_conf_4,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }

  if (max_active_blocks_conf_4 == max_active_blocks_conf_1 ||
      (max_active_blocks_conf_4 > 0 && rows <= sm_count)) {
    *success = true;
    return LaunchRmsNormBlockSMemImpl<LOAD,
                                      STORE,
                                      ComputeType,
                                      kPackSize,
                                      block_size_conf_4>(
        stream, load, store, smem, rows, cols, epsilon, col_divisor);
  }

  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        RmsNormBlockSMemImpl<LOAD,
                             STORE,
                             ComputeType,
                             kPackSize,
                             block_size_conf_3>,
        block_size_conf_3,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1 ||
      (max_active_blocks_conf_3 > 0 && rows <= sm_count)) {
    *success = true;
    return LaunchRmsNormBlockSMemImpl<LOAD,
                                      STORE,
                                      ComputeType,
                                      kPackSize,
                                      block_size_conf_3>(
        stream, load, store, smem, rows, cols, epsilon, col_divisor);
  }

  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        RmsNormBlockSMemImpl<LOAD,
                             STORE,
                             ComputeType,
                             kPackSize,
                             block_size_conf_2>,
        block_size_conf_2,
        smem);
    if (err != cudaSuccess) {
      return err;
    }
  }
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1 ||
      (max_active_blocks_conf_2 > 0 && rows <= sm_count)) {
    *success = true;
    return LaunchRmsNormBlockSMemImpl<LOAD,
                                      STORE,
                                      ComputeType,
                                      kPackSize,
                                      block_size_conf_2>(
        stream, load, store, smem, rows, cols, epsilon, col_divisor);
  }

  *success = true;
  return LaunchRmsNormBlockSMemImpl<LOAD,
                                    STORE,
                                    ComputeType,
                                    kPackSize,
                                    block_size_conf_1>(
      stream, load, store, smem, rows, cols, epsilon, col_divisor);
}

template <typename LOAD, typename STORE, typename ComputeType>
struct TryDispatchRmsNormBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream,
                         LOAD load,
                         STORE store,
                         const int32_t rows,
                         const int32_t cols,
                         const float epsilon,
                         ComputeType col_divisor,
                         bool* success) {
    if (cols % 4 == 0 && CanPackAs<LOAD>(load, 4) &&
        CanPackAs<STORE>(store, 4)) {
      return TryDispatchRmsNormBlockSMemImplBlockSize<LOAD,
                                                      STORE,
                                                      ComputeType,
                                                      4>(
          stream, load, store, rows, cols, epsilon, col_divisor, success);
    } else if (cols % 2 == 0 && CanPackAs<LOAD>(load, 2) &&
               CanPackAs<STORE>(store, 2)) {
      return TryDispatchRmsNormBlockSMemImplBlockSize<LOAD,
                                                      STORE,
                                                      ComputeType,
                                                      2>(
          stream, load, store, rows, cols, epsilon, col_divisor, success);
    } else {
      return TryDispatchRmsNormBlockSMemImplBlockSize<LOAD,
                                                      STORE,
                                                      ComputeType,
                                                      1>(
          stream, load, store, rows, cols, epsilon, col_divisor, success);
    }
  }
};

template <typename LOAD, typename STORE, typename ComputeType>
inline cudaError_t TryDispatchRmsNormBlockSMemImpl(cudaStream_t stream,
                                                   LOAD load,
                                                   STORE store,
                                                   const int32_t rows,
                                                   const int32_t cols,
                                                   const float epsilon,
                                                   ComputeType col_divisor,
                                                   bool* success) {
  return TryDispatchRmsNormBlockSMemImplPackSize<LOAD, STORE, ComputeType>()(
      stream, load, store, rows, cols, epsilon, col_divisor, success);
}

template <typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value,
                               cudaError_t>::type
DispatchRmsNorm(cudaStream_t stream,
                LOAD load,
                STORE store,
                const int32_t rows,
                const int32_t cols,
                const float epsilon) {
  const ComputeType col_divisor = 1.0f / cols;
  bool dispatch_smem_impl_success;
  {
    cudaError_t err = TryDispatchRmsNormBlockSMemImpl<LOAD, STORE, ComputeType>(
        stream,
        load,
        store,
        rows,
        cols,
        epsilon,
        col_divisor,
        &dispatch_smem_impl_success);
    if (err != cudaSuccess) {
      return err;
    }
  }
  return cudaSuccess;
}

template <typename SRC, typename DST>
struct SkipLoadAndStoreResidual {
  using LoadType = DST;
  // need to aseert SRC equals to DST.
  SkipLoadAndStoreResidual(SRC* src,
                           const SRC* bias,
                           const SRC* skip,
                           SRC* residual_bias_out,
                           float alpha,
                           int32_t row_size)
      : src(src),
        bias(bias),
        skip(skip),
        residual_bias_out(residual_bias_out),
        alpha(alpha),
        row_size(row_size) {}
  template <int N>
  __device__ void load(DST* dst, int32_t row, int32_t col) const {
    Pack<SRC, N> src_pack;
    Pack<SRC, N> bias_pack;
    Pack<SRC, N> skip_pack;
    Pack<DST, N> residual_out_pack;

    const int32_t offset = (row * row_size + col) / N;
    const int32_t bias_offset = col / N;
    src_pack = *(reinterpret_cast<const Pack<SRC, N>*>(src) + offset);
    bias_pack = *(reinterpret_cast<const Pack<SRC, N>*>(bias) + bias_offset);
    skip_pack = *(reinterpret_cast<const Pack<SRC, N>*>(skip) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      residual_out_pack.elem[i] =
          src_pack.elem[i] + bias_pack.elem[i] + skip_pack.elem[i];
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = residual_out_pack.elem[i];
    }
    *(reinterpret_cast<Pack<SRC, N>*>(residual_bias_out) + offset) =
        residual_out_pack;
  }

  SRC* src;
  const SRC* bias;
  const SRC* skip;
  SRC* residual_bias_out;
  float alpha;
  int32_t row_size;
};

template <typename SRC, typename DST>
struct AffineStore {
  AffineStore(DST* y, int32_t row_size, const DST* gamma, const DST* beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template <int N>
  __device__ void store(const SRC* src, int32_t row, int32_t col) {
    Pack<DST, N> y_pack;
    Pack<DST, N> gamma_pack;
    Pack<DST, N> beta_pack;
    const int32_t offset = (row * row_size + col) / N;
    const int32_t gamma_offset = col / N;
    gamma_pack = *(reinterpret_cast<const Pack<DST, N>*>(gamma) + gamma_offset);

    // Author(Zhengzekang): Bias maybe optional.
    if (beta) {
      beta_pack = *(reinterpret_cast<const Pack<DST, N>*>(beta) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; i++) {
        beta_pack.elem[i] = static_cast<DST>(0.0f);
      }
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      float normalized_i = static_cast<float>(src[i]);
      float normalized_val =
          normalized_i * static_cast<float>(gamma_pack.elem[i]) +
          static_cast<float>(beta_pack.elem[i]);
      y_pack.elem[i] = static_cast<DST>(normalized_val);
    }
    *(reinterpret_cast<Pack<DST, N>*>(y) + offset) = y_pack;
  }
  DST* y;
  int32_t row_size;
  const DST* gamma;
  const DST* beta;
};

// ======== For Int8 Output ========
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
                   const DST* gamma,
                   const DST* beta,
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
    Pack<DST, N> gamma_pack;
    Pack<DST, N> beta_pack;
    Pack<OutType, N> out_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    gamma_pack = *(reinterpret_cast<const Pack<DST, N>*>(gamma) + gamma_offset);

    // Author(Zhengzekang): Bias maybe optional.
    if (beta) {
      beta_pack = *(reinterpret_cast<const Pack<DST, N>*>(beta) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; i++) {
        beta_pack.elem[i] = static_cast<DST>(0.0f);
      }
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      float normalized_i = static_cast<float>(src[i]);
      float normalized_val =
          normalized_i * static_cast<float>(gamma_pack.elem[i]) +
          static_cast<float>(beta_pack.elem[i]);
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
  const DST* gamma;
  const DST* beta;
  const int quant_round_type;
  const float quant_out_scale;
  const float quant_max_bound;
  const float quant_min_bound;
};

#endif

}  // namespace

template <typename T, typename Context>
void RmsNormKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const paddle::optional<DenseTensor>& bias,
                   const paddle::optional<DenseTensor>& residual,
                   const DenseTensor& norm_weight,
                   const paddle::optional<DenseTensor>& norm_bias,
                   const float epsilon,
                   const int begin_norm_axis,
                   const float quant_scale,
                   const int quant_round_type,
                   const float quant_max_bound,
                   const float quant_min_bound,
                   DenseTensor* out,
                   DenseTensor* residual_out) {
#if defined(PADDLE_WITH_HIP)
  LOG(ERROR) << "Please compile with CUDA, ROCM platform isn't support it";
#else
  using ComputeType = typename phi::dtype::MPTypeTrait<T>::Type;

  const T* x_data = x.data<T>();
  const T* norm_weight_data = norm_weight.data<T>();
  const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;

  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }

  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  if (residual) {
    // Do RMSNorm(bias_add + residual + x)
    T* residual_out_data = dev_ctx.template Alloc<T>(residual_out);
    const T* residual_data = residual.get().data<T>();
    const T* bias_data = bias ? bias.get().data<T>() : nullptr;
    ResidualAddBiasLoad<T, ComputeType> load(
        x_data, residual_data, bias_data, residual_out_data, cols);
    if (quant_scale <= 0.0f) {
      // No Quantize.
      T* out_data = dev_ctx.template Alloc<T>(out);
      AffineStore<ComputeType, T> store(
          out_data, cols, norm_weight_data, norm_bias_data);
      DispatchRmsNorm<decltype(load), decltype(store), ComputeType>(
          dev_ctx.stream(), load, store, rows, cols, epsilon);
    } else {
      // Quantize and output int8.
      int8_t* out_data = dev_ctx.template Alloc<int8_t>(out);
      AffineQuantStore<int8_t, ComputeType, T, true, true> store(
          out_data,
          cols,
          norm_weight_data,
          norm_bias_data,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);

      DispatchRmsNorm<decltype(load), decltype(store), ComputeType>(
          dev_ctx.stream(), load, store, rows, cols, epsilon);
    }
  } else {
    DirectLoad<T, ComputeType> load(x_data, cols);
    if (quant_scale <= 0.0f) {
      // No Quantize.
      T* out_data = dev_ctx.template Alloc<T>(out);
      AffineStore<ComputeType, T> store(
          out_data, cols, norm_weight_data, norm_bias_data);
      DispatchRmsNorm<decltype(load), decltype(store), ComputeType>(
          dev_ctx.stream(), load, store, rows, cols, epsilon);
    } else {
      // Quantize and output int8.
      int8_t* out_data = dev_ctx.template Alloc<int8_t>(out);
      AffineQuantStore<int8_t, ComputeType, T, true, true> store(
          out_data,
          cols,
          norm_weight_data,
          norm_bias_data,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
      DispatchRmsNorm<decltype(load), decltype(store), ComputeType>(
          dev_ctx.stream(), load, store, rows, cols, epsilon);
    }
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(rms_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::RmsNormKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
