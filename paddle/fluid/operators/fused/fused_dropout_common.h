/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <cooperative_groups.h>
#include <cuda.h>
#include <curand_kernel.h>

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

#define MAX_CACHE_BYTES 16

/**
 * get the threads for fused_residual_dropout_bias:
 * 1D blocks: blockDim.x = cols
 * 2D grids: gridDim.y = rows
 */
inline platform::GpuLaunchConfig Get1DBlocksAnd2DGrids(
    const platform::CUDADeviceContext &ctx, const uint32_t rows,
    const uint32_t cols, const int VecSize) {
  const uint32_t tmp_cols = cols / VecSize;
  int threads = std::max(
      static_cast<uint32_t>(32),
      std::min(tmp_cols, static_cast<uint32_t>(ctx.GetMaxThreadsPerBlock())));
  const auto blocks_x =
      std::max(static_cast<uint32_t>(1), (tmp_cols + threads - 1) / threads);
  const auto blocks_y = std::max(static_cast<uint32_t>(1), rows);
  platform::GpuLaunchConfig config;
  config.block_per_grid.x = blocks_x;
  config.block_per_grid.y = blocks_y;
  config.thread_per_block.x = threads;
  return config;
}

__forceinline__ __device__ void Rand1(curandStatePhilox4_32_10_t *state,
                                      float *data) {
  data[0] = curand_uniform(state);
}

__forceinline__ __device__ void Rand2(curandStatePhilox4_32_10_t *state,
                                      float *data) {
  data[0] = curand_uniform(state);
  data[1] = curand_uniform(state);
}

__forceinline__ __device__ void Rand4(curandStatePhilox4_32_10_t *state,
                                      float *data) {
  float4 rand4 = curand_uniform4(state);
  data[0] = rand4.x;
  data[1] = rand4.y;
  data[2] = rand4.w;
  data[3] = rand4.z;
}

__forceinline__ __device__ void Rand8(curandStatePhilox4_32_10_t *state,
                                      float *data) {
  Rand4(state, data);
  Rand4(state, data + 4);
}

__forceinline__ __device__ void RandVec(curandStatePhilox4_32_10_t *state,
                                        float *data, const int VecSize) {
  if (VecSize == 1) {
    Rand1(state, data);
  } else if (VecSize == 2) {
    Rand2(state, data);
  } else if (VecSize == 4) {
    Rand4(state, data);
  } else if (VecSize == 8) {
    Rand8(state, data);
  } else {
    return;
  }
}

}  // namespace operators
}  // namespace paddle
