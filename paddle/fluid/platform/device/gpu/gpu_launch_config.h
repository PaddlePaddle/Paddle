// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

// Used for compute gpu launch parameter config

#pragma once

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/kernel_utils.h"

/* CUDA performs better as thread_per_block
   num is between [64, 512] */
#define PREDEFINED_BLOCK_SIZE 512

namespace paddle {
namespace platform {

template <typename T = int>
inline T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

/* https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
   for round integer value into next highest power of 2. */
static inline int RoundToPowerOfTwo(int n) {
  n--;
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::min(1024, std::max(32, (n + 1)));
}

#ifdef WITH_NV_JETSON
// The number of threads cannot be assigned 1024 in some cases when the device
// is nano or tx2 .
template <typename GPUContext>
inline void ChangeThreadNum(const GPUContext& context,
                            int* num_thread,
                            int alternative_num_thread = 512) {
  if (context.GetComputeCapability() == 53 ||
      context.GetComputeCapability() == 62) {
    *num_thread = alternative_num_thread;
  }
}
#endif

struct GpuLaunchConfig {
 public:
  GpuLaunchConfig() {}

  size_t GetThreadNum() const { return GetBlockSize() * GetGridSize(); }

  size_t GetGridSize() const {
    return block_per_grid.x * block_per_grid.y * block_per_grid.z;
  }

  size_t GetBlockSize() const {
    return thread_per_block.x * thread_per_block.y * thread_per_block.z;
  }

  bool IsEmpty() const {
    return phi::IsEmptyKernelDim(block_per_grid) ||
           phi::IsEmptyKernelDim(thread_per_block);
  }

  int compute_capability = 0;
  dim3 thread_per_block = dim3(1, 1, 1);
  dim3 block_per_grid = dim3(1, 1, 1);
};

/* According to NVIDIA, if number of threads per block is 64/128/256/512,
 * cuda performs better. And number of blocks should be greater (at least
 * 2x~4x) than number of SMs. Hence, SM count is took into account within
 * this function to determine the right number of threads per block. */
inline GpuLaunchConfig GetGpuLaunchConfig1D(const phi::GPUContext& context,
                                            int64_t numel,
                                            int vec_size = 1) {
  PADDLE_ENFORCE_GE(numel,
                    0,
                    platform::errors::InvalidArgument(
                        "element quantity should be greater than or equal 0,"
                        " but received value is: %d.",
                        numel));
  // Get compute_capability
  const int capability = context.GetComputeCapability();
  /* If thread number per block is 64/128/256/512, cuda performs better.*/
  int limit_threads =
      std::min(PREDEFINED_BLOCK_SIZE, context.GetMaxThreadsPerBlock());
#ifdef WITH_NV_JETSON
  if (capability == 53 || capability == 62) {
    limit_threads = 512;
  }
#endif
  int threads = limit_threads;
  int sm_count = context.GetSMCount();
  int64_t active_threads_num = numel / vec_size;
  if (active_threads_num / (sm_count << 1) < limit_threads) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about twice of SM, to acquire better performance.
    threads = RoundToPowerOfTwo(active_threads_num / (sm_count << 1));
  } else if (active_threads_num / (sm_count << 2) < limit_threads) {
    // Round up threads number into an exponential multiple of 2, while number
    // of acitve blocks is about 4 times of SM, to acquire better performance.
    threads = RoundToPowerOfTwo(active_threads_num / (sm_count << 2));
  }
  // Number of threads per block shall be larger than 64.
  threads = std::max(64, threads);
  int64_t blocks = DivUp<int64_t>(DivUp<int64_t>(numel, vec_size), threads);
  int limit_blocks = context.GetCUDAMaxGridDimSize()[0];
  if (blocks > limit_blocks) {
    blocks = limit_blocks;
  }

  GpuLaunchConfig config;
  config.thread_per_block.x = threads;
  config.block_per_grid.x = blocks;
  config.compute_capability = capability;
  return config;
}

inline GpuLaunchConfig GetGpuLaunchConfig2D(const phi::GPUContext& context,
                                            int64_t x_dim,
                                            int64_t y_dim) {
  PADDLE_ENFORCE_GT(
      x_dim,
      0,
      platform::errors::InvalidArgument("x dim number should greater than 0,"
                                        " but received value is: %d",
                                        x_dim));
  PADDLE_ENFORCE_GT(
      y_dim,
      0,
      platform::errors::InvalidArgument("y dim number should greater than 0,"
                                        " but received value is: %d",
                                        y_dim));

  const int kThreadsPerBlock = 256;
  // NOTE(zengjinle): cast std::min<int64_t> result to int is safe here, because
  // kThreadsPerBlock is always very small.
  int block_cols = std::min<int64_t>(x_dim, kThreadsPerBlock);
  int block_rows = std::max<int64_t>(kThreadsPerBlock / block_cols, 1);

  int max_physical_threads = context.GetMaxPhysicalThreadCount();
  const int max_blocks = (std::max)(max_physical_threads / kThreadsPerBlock, 1);

  GpuLaunchConfig config;
  // Noticed, block size is not align to 32, if needed do it yourself.
  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = std::min<int64_t>(DivUp<int64_t>(x_dim, block_cols), max_blocks);
  int grid_y = std::min<int64_t>(max_blocks / grid_x,
                                 std::max<int64_t>(y_dim / block_rows, 1));

  config.block_per_grid = dim3(grid_x, grid_y, 1);
  return config;
}

template <typename Context>
void LimitGridDim(const Context& ctx, dim3* grid_dim) {
  auto max_grid_dim =
      reinterpret_cast<const phi::GPUContext&>(ctx).GetCUDAMaxGridDimSize();
  grid_dim->x = grid_dim->x < max_grid_dim[0] ? grid_dim->x : max_grid_dim[0];
  grid_dim->y = grid_dim->y < max_grid_dim[1] ? grid_dim->y : max_grid_dim[1];
  grid_dim->z = grid_dim->z < max_grid_dim[2] ? grid_dim->z : max_grid_dim[2];
}
}  // namespace platform
}  // namespace paddle

#endif
