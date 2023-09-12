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

#pragma once

#if defined(PADDLE_WITH_CUDA)
#include <cooperative_groups.h>
#include <cuda.h>
#include <curand_kernel.h>
#endif

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/layer_norm_impl.cu.h"

namespace phi {
namespace fusion {

#define CACHE_LINE 128
#define MAX_CACHE_BYTES (CACHE_LINE / CHAR_BIT)

/**
 * get the threads for fused_residual_dropout_bias:
 * 1D blocks: blockDim.x = cols
 * 2D grids: gridDim.y = rows
 */
inline phi::backends::gpu::GpuLaunchConfig Get1DBlocksAnd2DGrids(
    const phi::GPUContext &ctx,
    const uint32_t rows,
    const uint32_t cols,
    const int vec_size) {
  const uint32_t tmp_cols = cols / vec_size;
  // NOTE(wangxi): We set max_block_size to 512, for `FusedResidualDropoutBias`
  // needs too many register resources. If data_type is float16, CUDA
  // error(701) will occur when block_size is 1024. Which error is
  // 'cudaErrorLaunchOutOfResources', this indicates that a launch did not
  // occur because it did not have appropriate resources.
  // Of course, this kernel can be optimized later to reduce the use
  // of registers.
  int threads = std::max(static_cast<uint32_t>(32),
                         std::min(tmp_cols,
                                  static_cast<uint32_t>(std::min(
                                      ctx.GetMaxThreadsPerBlock(), 512))));
  const auto blocks_x =
      std::max(static_cast<uint32_t>(1), (tmp_cols + threads - 1) / threads);
  const auto blocks_y = std::max(static_cast<uint32_t>(1), rows);
  phi::backends::gpu::GpuLaunchConfig config;
  config.block_per_grid.x = blocks_x;
  config.block_per_grid.y = blocks_y;
  config.thread_per_block.x = threads;
  return config;
}

template <int VecSize>
__forceinline__ __device__ void RandVec(curandStatePhilox4_32_10_t *state,
                                        float *data);

template <>
__forceinline__ __device__ void RandVec<1>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  data[0] = curand_uniform(state);
}

template <>
__forceinline__ __device__ void RandVec<2>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  data[0] = curand_uniform(state);
  data[1] = curand_uniform(state);
}

template <>
__forceinline__ __device__ void RandVec<4>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  float4 rand4 = curand_uniform4(state);
  data[0] = rand4.x;
  data[1] = rand4.y;
  data[2] = rand4.w;
  data[3] = rand4.z;
}

template <>
__forceinline__ __device__ void RandVec<8>(curandStatePhilox4_32_10_t *state,
                                           float *data) {
  RandVec<4>(state, data);
  RandVec<4>(state, data + 4);
}

template <typename T>
inline void SetZero(const phi::GPUContext &ctx, T *ptr, const size_t size) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(ptr, 0, size * sizeof(T), ctx.stream()));
}

/**
 * reduce the sum of 128 cols data by 8*VecSize warps
 **/
template <typename T, int VecSize, int BlockSizeX, int BlockSizeY>
inline __device__ void CalculateDBias(const T *tmp_sum,
                                      T *dbias,
                                      const int cols) {
  // save temporary sum to cache and do transpose
  __shared__ T cache[BlockSizeX * VecSize][BlockSizeY];
  for (int i = 0; i < VecSize; i++) {
    cache[threadIdx.x * VecSize + i][threadIdx.y] = tmp_sum[i];
  }
  __syncthreads();
  // reduce sum
  T sum[2] = {static_cast<T>(0)};
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int x = tid >> 5;  // warp id
  int y = tid & 31;  // thread id on warp 0~31

  // need BlockSizeX * VecSize warps
  for (int j = x; j < BlockSizeX * VecSize; j += 32) {
// reduce 128 to 32
#pragma unroll
    for (int i = 0; i < (BlockSizeY >> 5); i++) {
      sum[(j >> 5)] += cache[j][y + i * 32];
    }
  }

  int reduce_num_pre_thread = (BlockSizeX * VecSize + 31) / 32;
  // reduce 32 to 1
  for (int i = 0; i < reduce_num_pre_thread; i++) {
    sum[i] = phi::funcs::WarpReduceSum(sum[i]);
  }

  // save sum to dbias
  if (y == 0 && x < BlockSizeX * VecSize) {
    for (int i = 0; i < reduce_num_pre_thread; i++) {
      int bias_id = blockIdx.x * BlockSizeX * VecSize + x + i * 32;
      if (bias_id < cols) {
        dbias[bias_id] = sum[i];
      }
    }
  }
}

template <typename T>
inline __device__ T GetFactor(const float dropout_prob,
                              const bool is_upscale_in_train,
                              const bool is_test) {
  T factor = is_upscale_in_train ? static_cast<T>(1.0f / (1.0f - dropout_prob))
                                 : static_cast<T>(1.0f);
  if (is_test) {
    factor = is_upscale_in_train ? static_cast<T>(1.0f)
                                 : static_cast<T>(1.0f - dropout_prob);
  }
  return factor;
}

}  // namespace fusion
}  // namespace phi
