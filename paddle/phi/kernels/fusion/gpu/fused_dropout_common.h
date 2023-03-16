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

}  // namespace fusion
}  // namespace phi
