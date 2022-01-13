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

#ifdef __HIPCC__
// HIP results in error or nan if > 256
#define PREDEFINED_BLOCK_SIZE 256
#else
/* CUDA performs better as thread_per_block
   num is between [64, 512] */
#define PREDEFINED_BLOCK_SIZE 512
#endif

namespace paddle {
namespace platform {

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

/* https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
   for round integer value into next highest power of 2. */
static inline int RoundToPowerOfTwo(int n) {
  n--;
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
#ifdef __HIPCC__
  return std::min(256, std::max(32, (n + 1)));
#else
  return std::min(1024, std::max(32, (n + 1)));
#endif
}

#ifdef WITH_NV_JETSON
// The number of threads cannot be assigned 1024 in some cases when the device
// is nano or tx2 .
inline void ChangeThreadNum(const platform::CUDADeviceContext& context,
                            int* num_thread, int alternative_num_thread = 512) {
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

  int compute_capability = 0;
  dim3 thread_per_block = dim3(1, 1, 1);
  dim3 block_per_grid = dim3(1, 1, 1);
};

/* According to NVIDIA, if number of threads per block is 64/128/256/512,
  * cuda performs better. And number of blocks should be greater (at least
  * 2x~4x) than number of SMs. Hence, SM count is took into account within
  * this function to determine the right number of threads per block. */
inline GpuLaunchConfig GetGpuLaunchConfig1D(
    const platform::CUDADeviceContext& context, int64_t numel,
    int vec_size = 1) {
  PADDLE_ENFORCE_GT(numel, 0, platform::errors::InvalidArgument(
                                  "element quantity should be greater than 0,"
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
  int active_threads_num = numel / vec_size;
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
  int blocks = DivUp(DivUp(numel, vec_size), threads);

  GpuLaunchConfig config;
  config.thread_per_block.x = threads;
  config.block_per_grid.x = blocks;
  config.compute_capability = capability;
  return config;
}

inline GpuLaunchConfig GetGpuLaunchConfig2D(
    const platform::CUDADeviceContext& context, int x_dim, int y_dim) {
  PADDLE_ENFORCE_GT(x_dim, 0, platform::errors::InvalidArgument(
                                  "x dim number should greater than 0,"
                                  " but received value is: %d",
                                  x_dim));
  PADDLE_ENFORCE_GT(y_dim, 0, platform::errors::InvalidArgument(
                                  "y dim number should greater than 0,"
                                  " but received value is: %d",
                                  y_dim));

  const int kThreadsPerBlock = 256;
  int block_cols = (std::min)(x_dim, kThreadsPerBlock);
  int block_rows = (std::max)(kThreadsPerBlock / block_cols, 1);

  int max_physical_threads = context.GetMaxPhysicalThreadCount();
  const int max_blocks = (std::max)(max_physical_threads / kThreadsPerBlock, 1);

  GpuLaunchConfig config;
  // Noticed, block size is not align to 32, if needed do it yourself.
  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = (std::min)(DivUp(x_dim, block_cols), max_blocks);
  int grid_y =
      (std::min)(max_blocks / grid_x, (std::max)(y_dim / block_rows, 1));

  config.block_per_grid = dim3(grid_x, grid_y, 1);
  return config;
}

// TODO(wangchaochaohu): 3D will add later

}  // namespace platform
}  // namespace paddle

#endif
