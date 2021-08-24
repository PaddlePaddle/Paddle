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

#include <iostream>
#include <memory>

#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

/**
 * get 1D threads and blocks
 */
template <int VecSize = 4>
inline std::pair<uint32_t, uint32_t> Get1DThreadsAndBlocks(
    const platform::CUDADeviceContext &ctx, const uint64_t n) {
  const uint64_t tmp_n = n / VecSize;
  int threads = std::max(
      (uint64_t)32, std::min(tmp_n, (uint64_t)ctx.GetMaxThreadsPerBlock()));
  int blocks = std::max((uint64_t)1, (tmp_n + threads - 1) / threads);
  return std::pair<uint32_t, uint32_t>{threads, blocks};
}

/**
 * get the threads for fused_residual_dropout_bias:
 * 1D blocks: blockDim.x = cols
 * 2D grids: gridDim.y = rows
 */
template <int VecSize = 4>
inline std::pair<dim3, dim3> Get1DBlocksAnd2DGrids(
    const platform::CUDADeviceContext &ctx, const uint32_t rows,
    const uint32_t cols) {
  const uint32_t tmp_cols = cols / VecSize;
  int threads = std::max(
      (uint32_t)32, std::min(tmp_cols, (uint32_t)ctx.GetMaxThreadsPerBlock()));
  int blocks_x = std::max((uint32_t)1, (tmp_cols + threads - 1) / threads);
  int blocks_y = std::max((uint32_t)1, rows);
  dim3 block_dim(threads, 1, 1);
  dim3 grid_dim(blocks_x, blocks_y, 1);
  return std::pair<dim3, dim3>{block_dim, grid_dim};
}

// aligned vector generates vectorized load/store on CUDA
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignedVector {
  T val[VecSize];
};

}  // namespace operators
}  // namespace paddle
