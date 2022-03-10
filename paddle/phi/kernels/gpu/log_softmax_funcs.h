// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cmath>
#include <limits>
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, int KernelWarpSize>
__device__ __forceinline__ T WarpReduceSum(T value) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
    T sum_val = paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, value, offset);
    value = value + sum_val;
  }
  return value;
}

template <typename T, int KernelWarpSize>
__device__ __forceinline__ T WarpReduceMax(T value) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
    T max_val = paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, value, offset);
    value = max(value, max_val);
  }
  return value;
}

// Returns the final item after reduce operation along block.x.
// Firstly, get shared memory(smem) offset, find the starting position for every
// y.
// Secondly, initialise every smem position with value 'val' of thread itself.
// Thirdly, apply standard reduction along x direction as below:
//
//   -> x direction
// [o o o o o o o o]    time 0
//  |     |/     /
//  |    /|    /
//  |  /  |  /
//  |/    |/
// [o o o o x x x x]    time 1
//  | |/ /
//  |/|/
// [o o x x x x x x]    time 2
//  |/
// [o x x x x x x x]    time 3
//
// Finally, return the first item.
// Imaging multiple reductions executed in paralell along y axis,
// Note that when blockDim.x is not 1, it's a EVEN number in all cases,
// and the size of shared memory is even as well.
template <typename T, template <typename> class Functor>
__forceinline__ __device__ T BlockReduceAlongDimX(T *shared, T val) {
  Functor<T> func;
  // This reduction is not Block-wise reduction, only reduce along block.x.
  // therefore the shared mem has offsets for different block.y.
  shared += threadIdx.y * blockDim.x;
  shared[threadIdx.x] = val;
  int offset = blockDim.x / 2;

  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset) {
      shared[threadIdx.x] =
          func(shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    offset /= 2;
  }
  __syncthreads();
  return shared[0];
}

// block.y covers inner_size. Threads along the x axis process dim_size
// elements, and make sure not to exceed the 1024 threads per block.
// Note that dim_threads namely blockDim.x is either 1 or a even number.
inline dim3 GetBlockSize(int dim_size, int inner_size) {
  int inner_threads = inner_size;
  inner_threads = std::min(inner_threads, 1024);
  int dim_threads = 1;

  while (dim_threads * inner_threads <= 1024 && dim_threads <= dim_size) {
    dim_threads *= 2;
  }
  dim_threads /= 2;
  return dim3(dim_threads, inner_threads);
}

// First cover the y axis as many blocks as possible.
// Then cover the x axis as many blocks as possible,
// and make sure not to exceed the max_active_blocks.
inline dim3 GetGridSize(dim3 block,
                        int max_active_blocks,
                        int outer_size,
                        int dim_size,
                        int inner_size) {
  int inner_blocks = (inner_size + block.y - 1) / block.y;
  if (inner_blocks > max_active_blocks) inner_blocks = max_active_blocks;

  int outer_blocks = (max_active_blocks + inner_blocks - 1) / inner_blocks;
  if (outer_blocks > outer_size) outer_blocks = outer_size;
  return dim3(outer_blocks, inner_blocks);
}

// When designing grid size and block size, priority is given to block size,
// and grid will be determined according to the maximum number of active blocks,
// which is set by as a experience value.
template <typename T, typename Kernel>
void ComputeLaunchConfigure(Kernel k,
                            int outer_size,
                            int dim_size,
                            int inner_size,
                            dim3 &grid,       // NOLINT
                            dim3 &block,      // NOLINT
                            int &shared_mem,  // NOLINT
                            int num_sm) {
  block = GetBlockSize(dim_size, inner_size);
  int block_threads = block.x * block.y;
  shared_mem = block.x == 1 ? 0 : block_threads * sizeof(T);
  int max_active_blocks = num_sm * 2;
  grid =
      GetGridSize(block, max_active_blocks, outer_size, dim_size, inner_size);
}

template <typename T, typename AccT, int NearGreaterPowerOfTwo>
__global__ void ComputeLogSoftmaxForwardInWarp(T *dst,
                                               const T *src,
                                               int batch_size,
                                               int element_count) {
  constexpr int near_greater_power_of_two = NearGreaterPowerOfTwo;
  constexpr int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  constexpr int warp_iter = near_greater_power_of_two / kernel_warp_size;
  int batch_id = blockDim.y * blockIdx.x + threadIdx.y;

  int thread_in_warp_idx = threadIdx.x;

  // 1.read data from global memory to registers
  AccT elements[warp_iter];  // NOLINT
  // set effective_element_count as the num of elements when warps do effective
  // work
  // set effective_element_count as 0, when warps do ineffective work
  int effective_element_count = (batch_id < batch_size) ? element_count : 0;
  for (int it = 0; it < warp_iter; ++it) {
    int element_index = thread_in_warp_idx + it * kernel_warp_size;
    if (element_index < effective_element_count) {
      elements[it] =
          static_cast<AccT>(src[batch_id * element_count + element_index]);
    } else {
      elements[it] = -std::numeric_limits<AccT>::infinity();
    }
  }

  // 2.compute max_value. For each thread, loop all registers to find max
  AccT max_value = elements[0];
#pragma unroll
  for (int it = 1; it < warp_iter; ++it) {
    max_value = (max_value > elements[it]) ? max_value : elements[it];
  }
  max_value = WarpReduceMax<AccT, kernel_warp_size>(max_value);

  // 3.For each warp, accumulate all thread registers
  AccT sum = 0.0f;
#pragma unroll
  for (int it = 0; it < warp_iter; ++it) {
    sum += std::exp(elements[it] - max_value);
  }
  sum = WarpReduceSum<AccT, kernel_warp_size>(sum);

  // 4.store result.
  sum = std::log(sum);
#pragma unroll
  for (int it = 0; it < warp_iter; ++it) {
    int element_index = thread_in_warp_idx + it * kernel_warp_size;
    if (element_index < effective_element_count) {
      dst[batch_id * element_count + element_index] =
          static_cast<T>(elements[it] - max_value - sum);
    } else {
      break;
    }
  }
}

}  // namespace phi
