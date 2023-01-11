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

#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"

#ifdef __HIPCC__
// HIP results in error or nan if > 256
#define PREDEFINED_BLOCK_SIZE 256
#else
// CUDA performs better when thread_per_block is between [64, 512]
#define PREDEFINED_BLOCK_SIZE 512
#endif

namespace phi {
namespace backends {
namespace gpu {

// Limitation of the setting in one dimension of cuda grid.
constexpr int kMultiDimslimit = 65536;

template <typename T = int64_t>
inline T DivUp(T a, T b) {
  return (a + b - 1) / b;
}

// https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
//   for round integer value into next highest power of 2.
inline int64_t RoundToNextHighPowOfTwo(int64_t n, int64_t min_val = 1) {
  n--;
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(min_val, (n + 1));
}

inline int64_t RoundToPowerOfTwo(int64_t n) {
  constexpr int64_t min_val = 32;
  int64_t num = RoundToNextHighPowOfTwo(n, min_val);
#ifdef __HIPCC__
  int64_t max_val = 256;
#else
  int64_t max_val = 1024;
#endif
  return std::min(max_val, num);
}

#ifdef WITH_NV_JETSON
// The number of threads cannot be assigned 1024 in some cases when the device
// is nano or tx2 .
inline void ChangeThreadNum(const phi::GPUContext& context,
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
                    phi::errors::InvalidArgument(
                        "numel is expected to be greater than or equal 0,"
                        " but received %d.",
                        numel));
  PADDLE_ENFORCE_GE(
      vec_size,
      1,
      phi::errors::InvalidArgument(
          "vec_size is expected greater than 0, but received %d.", vec_size));
  // Get compute_capability
  const int capability = context.GetComputeCapability();
  // If thread number per block is 64/128/256/512, cuda performs better.
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
  int blocks = DivUp<int64_t>(DivUp<int64_t>(numel, vec_size), threads);
  int limit_blocks = context.GetCUDAMaxGridDimSize()[0];
  if (blocks > limit_blocks) {
    blocks = limit_blocks;
  }

  GpuLaunchConfig config;
  config.thread_per_block.x = threads;
  config.block_per_grid.x = blocks;
  config.compute_capability = capability;

  VLOG(3) << "Get 1-D launch config: numel=" << numel
          << ", vec_size=" << vec_size << ", block_size=" << threads
          << ", grid_size=" << blocks << ", limit_blocks=" << limit_blocks
          << ", limit_threads=" << limit_threads;
  return config;
}

inline GpuLaunchConfig GetGpuLaunchConfig2D(const phi::GPUContext& context,
                                            int64_t x_dim,
                                            int64_t y_dim) {
  PADDLE_ENFORCE_GT(
      x_dim,
      0,
      phi::errors::InvalidArgument("x dim number should greater than 0,"
                                   " but received value is: %d",
                                   x_dim));
  PADDLE_ENFORCE_GT(
      y_dim,
      0,
      phi::errors::InvalidArgument("y dim number should greater than 0,"
                                   " but received value is: %d",
                                   y_dim));

  const int kThreadsPerBlock = 256;
  int block_cols = std::min<int64_t>(x_dim, kThreadsPerBlock);
  int block_rows = std::max(kThreadsPerBlock / block_cols, 1);

  int max_physical_threads = context.GetMaxPhysicalThreadCount();
  const int max_blocks = std::max(max_physical_threads / kThreadsPerBlock, 1);

  GpuLaunchConfig config;
  // Noticed, block size is not align to 32, if needed do it yourself.
  config.thread_per_block = dim3(block_cols, block_rows, 1);

  int grid_x = std::min<int64_t>(DivUp<int64_t>(x_dim, block_cols), max_blocks);
  int grid_y = std::min<int64_t>(max_blocks / grid_x,
                                 std::max<int64_t>(y_dim / block_rows, 1));

  config.block_per_grid = dim3(grid_x, grid_y, 1);
  return config;
}

static inline int GetLastPow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

inline GpuLaunchConfig GetGpuLaunchConfig3D(const phi::GPUContext& context,
                                            int num_img,
                                            int height,
                                            int width) {
  const int kThreadsPerBlock = 256;
  int max_threads_per_block = context.GetMaxThreadsPerBlock();  // 1024
  int max_threads = std::min(kThreadsPerBlock, max_threads_per_block);

  int block_x = std::min(GetLastPow2(width), max_threads);
  int block_y = std::min(GetLastPow2(height), max_threads / block_x);
  int block_z = std::min(num_img, max_threads / block_x / block_y);

  std::array<int, 3> max_grid_dim = context.GetCUDAMaxGridDimSize();
  int grid_x = std::min(max_grid_dim[0], DivUp<int>(width, block_x));
  int grid_y = std::min(max_grid_dim[1], DivUp<int>(height, block_y));
  int grid_z = std::min(max_grid_dim[2], DivUp<int>(num_img, block_z * 4));

  const int capability = context.GetComputeCapability();
  GpuLaunchConfig config;
  config.compute_capability = capability;
  config.thread_per_block = dim3(block_x, block_y, block_z);
  config.block_per_grid = dim3(grid_x, grid_y, grid_z);
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
}  // namespace gpu
}  // namespace backends
}  // namespace phi

#endif
