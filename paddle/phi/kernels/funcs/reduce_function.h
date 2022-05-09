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

// CUDA, XPU and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_XPU_KP)

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <vector>

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#ifndef PADDLE_WITH_XPU_KP
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#endif

#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#include "paddle/utils/string/string_helper.h"

// Reduce split or not, Whether to use ReduceHigherDim
#define REDUCE_SPLIT_BOUNDARY 512
#define REDUCE_VEC_SIZE 4

namespace kps = phi::kps;
#ifdef PADDLE_WITH_XPU_KP
using dim3 = phi::kps::dim3;
#endif
namespace phi {
namespace funcs {

namespace details {

static inline int GetLastPow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

static inline int64_t AlignUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// get strides of x_dim, reduce_dim and left_dim for reduceLastDim and reduceAny
static inline std::vector<int> GetDimStrides(const std::vector<int>& dims,
                                             const std::vector<int>& idx) {
  int n = static_cast<int>(idx.size());
  if (n == 0) return std::vector<int>();
  std::vector<int> strides(n);
  strides.back() = 1;
  for (int i = n - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[idx[i + 1]];
  }
  return strides;
}

#ifndef PADDLE_WITH_XPU_KP
// get blockDim for reduceLastDim and reduceAny
static inline int GetBlockDim(int block_dim) {
  return block_dim >= kps::details::kReduceMaxThread
             ? kps::details::kReduceMaxThread
             : GetLastPow2(block_dim);
}
#endif

// check reduce rand is valid
static inline void CheckReduceRank(int reduce_rank, int rank) {
  if (rank % 2 == 0) {
    PADDLE_ENFORCE_EQ(reduce_rank,
                      rank / 2,
                      phi::errors::InvalidArgument(
                          "ReduceOp: invalid reduce rank. When rank = %d, "
                          "reduce_rank must be %d, but got %d.",
                          rank,
                          rank / 2,
                          reduce_rank));
  } else {
    auto lower_rank = (rank - 1) / 2;
    auto upper_rank = (rank + 1) / 2;
    PADDLE_ENFORCE_EQ(
        reduce_rank == lower_rank || reduce_rank == upper_rank,
        true,
        phi::errors::InvalidArgument(
            "ReduceOp: invalid reduce rank. When rank = %d, reduce_rank "
            "must be %d or %d, but got %d.",
            rank,
            lower_rank,
            upper_rank,
            reduce_rank));
  }
}

// convert dims from vector to array
template <typename T, size_t ElementCount, typename VectorLikeType>
static inline phi::Array<T, ElementCount> VectorToArray(
    const VectorLikeType& vec) {
  PADDLE_ENFORCE_LE(
      vec.size(),
      ElementCount,
      phi::errors::InvalidArgument("Cub reduce Array: size not match. Received "
                                   "vec.size() %d > ElementCount %d.",
                                   vec.size(),
                                   ElementCount));
  size_t n = static_cast<size_t>(vec.size());
  phi::Array<T, ElementCount> ret;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = vec[i];
  }
  return ret;
}

static inline std::vector<int> GetReduceDim(const std::vector<int64_t>& dims,
                                            int dim_size,
                                            bool reduce_all) {
  std::vector<int> reduce_dims;
  if (reduce_all) {
    reduce_dims.resize(dim_size);
    int reduce_size = reduce_dims.size();
    for (int i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = i;
    }
  } else {
    for (auto e : dims) {
      PADDLE_ENFORCE_LT(e,
                        dim_size,
                        phi::errors::InvalidArgument(
                            "ReduceOp: invalid axis, when x_dims is %d, "
                            "axis[i] should less than x_dims, but got %d.",
                            dim_size,
                            e));
      reduce_dims.push_back(e >= 0 ? e : e + dim_size);
    }
  }
  return reduce_dims;
}

}  // namespace details

constexpr int kMaxRank = phi::DDim::kMaxRank;

enum ReduceType {
  kReduceLastDim = 0x01,    // when reduce_dim[0] == x_dim.size() - 1;
  kReduceHigherDim = 0x02,  // ReduceFirstDim or reduceSecondDim
  kReduceAny = 0x03,        // when reduce_dim.size() > 1
};

struct IndexCalculator {
  IndexCalculator(int dim,
                  const std::vector<int>& cal_dims,
                  const std::vector<int>& cal_strides,
                  const std::vector<int>& full_strides)
      : dim(dim) {
    dims = details::VectorToArray<int, kMaxRank>(cal_dims);
    strides = details::VectorToArray<int, kMaxRank>(full_strides);
    reduce_strides = details::VectorToArray<int, kMaxRank>(cal_strides);
#ifndef PADDLE_WITH_XPU_KP
    std::vector<kps::details::FastDivMod> cal_divmoders;
    // fast divmod
    for (auto i : cal_strides) {
      cal_divmoders.push_back(kps::details::FastDivMod(i));
    }
    divmoders = details::VectorToArray<kps::details::FastDivMod, kMaxRank>(
        cal_divmoders);
#endif
  }

  __device__ inline int operator()(int offset) const {
#ifdef PADDLE_WITH_XPU_KP
    int index = 0;
#pragma unroll
    for (int i = 0; i < kMaxRank; ++i) {
      if (i == dim) {
        break;
      }
      index += (offset / reduce_strides[i]) * strides[dims[i]];
      offset = offset % reduce_strides[i];
    }
    return index;
#else
    int index = 0;
#pragma unroll
    for (int i = 0; i < kMaxRank; ++i) {
      if (i == dim) {
        break;
      }
      auto divmod = divmoders[i].Divmod(offset);
      index += (divmod.val[0] * strides[dims[i]]);
      offset = divmod.val[1];
    }
    return index;
#endif
  }

  int dim;
  phi::Array<int, kMaxRank> dims;
  phi::Array<int, kMaxRank> strides;
  phi::Array<int, kMaxRank> reduce_strides;
#ifndef PADDLE_WITH_XPU_KP
  phi::Array<kps::details::FastDivMod, kMaxRank> divmoders;
#endif
};

template <bool ReduceLastDim = false>
struct ReduceIndexMapping {
  const kps::DimConfig dim;
  HOSTDEVICE explicit ReduceIndexMapping(const kps::DimConfig& dims)
      : dim(dims) {}

#ifdef PADDLE_WITH_XPU_KP
  __device__ __forceinline__ int BlockIdX() {
    if (ReduceLastDim) {
      return (cluster_id() / dim.split_num_x % dim.split_num_y);
    } else {
      return cluster_id() % dim.split_num_x;
    }
  }

  __device__ __forceinline__ int BlockIdY() {
    if (ReduceLastDim) {
      return (cluster_id() % dim.split_num_x);
    } else {
      return (cluster_id() / dim.split_num_x % dim.split_num_y);
    }
  }

  __device__ __forceinline__ int BlockDimX() { return dim.deal_size_x; }

  __device__ __forceinline__ int BlockDimY() { return 1; }

  __device__ __forceinline__ int GridDimX() {
    if (ReduceLastDim) {
      return dim.split_num_y;
    } else {
      return dim.split_num_x;
    }
  }

  __device__ __forceinline__ int GridDimY() {
    if (ReduceLastDim) {
      return dim.split_num_x;
    } else {
      return dim.split_num_y;
    }
  }

  __device__ __forceinline__ int GetLoopSize() {
    if (ReduceLastDim) {
      return dim.deal_size_y;
    } else {
      return dim.deal_size_x;
    }
  }
#else
  __device__ __forceinline__ int BlockIdX() { return blockIdx.x; }

  __device__ __forceinline__ int BlockIdY() { return blockIdx.y; }

  __device__ __forceinline__ int BlockDimX() { return blockDim.x; }

  __device__ __forceinline__ int BlockDimY() { return blockDim.y; }

  __device__ __forceinline__ int GridDimX() { return gridDim.x; }

  __device__ __forceinline__ int GridDimY() { return gridDim.y; }

  __device__ int GetLoopSize() { return 1; }
#endif
};

// when reduce_type == kReduceLastDim this struct will be used
// for higher performance
struct OneDimIndexCal {
  explicit OneDimIndexCal(int num) : stride(num) {}

  __device__ inline int operator()(int index) const { return index * stride; }
  int stride;
};

// reduce config
template <typename Ty>
struct ReduceConfig {
  ReduceConfig(const std::vector<int>& origin_reduce_dims,
               const std::vector<int>& origin_x_dim)
      : reduce_dims_origin(origin_reduce_dims), x_dim(origin_x_dim) {}

  // get the parameters of reduceKernel
  void Run(const KPDevice& dev_ctx) {
    // step1: update the reduce_dim left_dim and x_dim
    SetReduceDim();

    // step2: get the strides of dim for reduceAny and reduceLastDim
    SetStrides();

    // step3: get the type of reduce
    SetReduceType();

    // step4: set the block and grid for launch kernel
    SetBlockDim();
#ifndef PADDLE_WITH_XPU_KP
    // step5: limit the grid to prevent thead overflow
    paddle::platform::LimitGridDim(dev_ctx, &grid);
#endif
  }

  // when should_reduce_again is true, we need malloc temp space for temp data
  void SetOutputData(Ty* y_data,
                     const KPDevice& dev_ctx,
                     phi::DenseTensor* tmp) {
    if (should_reduce_again) {
      tmp->Resize(phi::make_ddim(
          {static_cast<int64_t>(left_num * grid.z * grid.y * sizeof(Ty))}));
      output_data = dev_ctx.Alloc<Ty>(tmp);
    } else {
      output_data = y_data;
    }
  }

 private:
  // set reduce_dim, left_dim and update x_dim
  // eg: x_dim = [2, 4, 6] origin_reduce_dims = [0, 1]
  //     --SetReduceDim--> x_dim = [8,6], reduce_dim = [0], left_dim = [1]
  void SetReduceDim() {
    std::set<int> reduce_set;
    for (auto e : reduce_dims_origin) {
      auto pos = e >= 0 ? e : e + x_dim.size();
      reduce_set.insert(pos);
    }

    std::vector<int> reduce_dim_temp(reduce_set.begin(), reduce_set.end());
    std::sort(reduce_dim_temp.begin(), reduce_dim_temp.end());

    // update reduce_dim and x_dim
    std::vector<int> x_new_dim;

    reduce_dim.push_back(reduce_dim_temp[0]);
    x_new_dim.push_back(x_dim[0]);

    int idx_reduce = 1;
    int num = 0;

    if (reduce_dim_temp.size() > 1) {
      for (int i = 1; i < x_dim.size(); i++) {
        if ((idx_reduce < reduce_dim_temp.size()) &&
            (i == reduce_dim_temp[idx_reduce])) {
          int result =
              reduce_dim_temp[idx_reduce] - reduce_dim[reduce_dim.size() - 1];
          bool is_equal = ((result - num) == 1);
          if (is_equal) {
            x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
            num++;
          } else {
            reduce_dim.push_back(reduce_dim_temp[idx_reduce] - num);
            x_new_dim.push_back(x_dim[i]);
          }
          idx_reduce++;
        } else {
          x_new_dim.push_back(x_dim[i]);
        }
      }
    } else {
      x_new_dim = x_dim;
    }

    // update x_dim
    x_dim = x_new_dim;
    std::vector<int>().swap(x_new_dim);

    std::vector<int> reduce_dim_new;
    int is_reduced = 0;
    for (auto e : reduce_dim) {
      is_reduced |= 1 << e;
    }

    std::vector<int>().swap(reduce_dim);

    for (int i = 0; i < x_dim.size(); i++) {
      if ((i == 0) || (((is_reduced >> i) ^ (is_reduced >> (i - 1))) & 1)) {
        x_new_dim.push_back(x_dim[i]);
        if ((is_reduced >> i) & 1)
          reduce_dim_new.push_back(x_new_dim.size() - 1);
      } else {
        x_new_dim[x_new_dim.size() - 1] *= x_dim[i];
      }
    }

    x_dim = x_new_dim;
    reduce_dim = reduce_dim_new;

    int x_rank = static_cast<int>(x_dim.size());
    std::set<int> left_set;

    for (int i = 0; i < x_rank; ++i) {
      left_set.insert(i);
    }

    for (auto e : reduce_dim) {
      left_set.erase(e);
    }

    left_dim.assign(left_set.begin(), left_set.end());

    // if the last dim gets involved in reduction
    reduce_last_dim = (reduce_dim.back() == x_dim.size() - 1);
  }

  // set x_strides, reduce_strides, left_strides for reduceLastDim and reduceAny
  // eg: x_dim = [8, 6], reduce_dim = [0], left_dim = [1]
  //     --SetStrides--> x_strides= [6,1], reduce_strides = [1],
  //     left_strides = [1]
  void SetStrides() {
    std::vector<int> idx_dim;
    for (int i = 0; i < x_dim.size(); i++) {
      idx_dim.push_back(i);
    }

    x_strides = details::GetDimStrides(x_dim, idx_dim);
    reduce_strides = details::GetDimStrides(x_dim, reduce_dim);
    left_strides = details::GetDimStrides(x_dim, left_dim);
    reduce_num = reduce_strides[0] * x_dim[reduce_dim[0]];

    left_num = 1;
    if (left_dim.size()) {
      left_num = left_strides[0] * x_dim[left_dim[0]];
    }
  }

  // get the reduceType
  // eg: x_dim = [8, 6] reduce_dim = [0] --> ReduceHigherDim -->reduceFirstDim
  //     x_dim = [8, 6] reduce_dim = [1] --> reduceLastDim
  //     x_dim = [8] reduce_dim = [0] --> reduceAll
  //     x_dim = [8, 6, 4, 2] reduce_dim = [0, 2] --> reduceAny
  void SetReduceType() {
    int rank = x_dim.size();
    int reduce_rank = reduce_dim.size();
#ifdef PADDLE_WITH_XPU_KP
    bool not_higher = x_dim[0] > 1;
#else
    int device_id = paddle::platform::GetCurrentDeviceId();
    int max_grid_z = phi::backends::gpu::GetGpuMaxGridDimSize(device_id)[2];
    bool not_higher = x_dim[0] >= max_grid_z;
#endif
    if (reduce_last_dim && (reduce_rank == 1)) {
      reduce_type = static_cast<int>(ReduceType::kReduceLastDim);
    } else if (reduce_rank == 1) {
      reduce_type = static_cast<int>(ReduceType::kReduceHigherDim);
      if (rank == 3 && not_higher) {
        reduce_type = static_cast<int>(ReduceType::kReduceAny);
      }
    } else {
      reduce_type = static_cast<int>(ReduceType::kReduceAny);
    }
  }

#ifndef PADDLE_WITH_XPU_KP
  void SetBlockDimForReduceAny(dim3* block_dim, dim3* grid_dim) {
    constexpr int min_reduce_num_per_thread = 16;
    constexpr int max_reduce_num_per_thread = 256;
    constexpr int max_num_threads = kps::details::kReduceMaxThread;

    // set block size.
    // 1. If reduce_last_dim == true, all the threads whose threadIdx.y are same
    //    will process the reduction for one output.
    //    The number of output for one block is blockDim.y;
    // 2. If reduce_last_dim == false, different threadIdx.x will process
    //    different reduction and gets the output separately. If it is
    //    necessary, it should reduce in block y.
    //    The number of output for one block is blockDim.x;
    int block_x, block_y;
    int grid_num, reduce_num_per_thread;
    if (reduce_last_dim) {
      block_x = details::GetBlockDim(reduce_num);
      block_y = details::GetBlockDim(left_num);
      block_dim->x = block_x;
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      grid_num = details::AlignUp(left_num, block_dim->y);
      reduce_num_per_thread = details::AlignUp(reduce_num, block_dim->x);
    } else {
      block_x = details::GetBlockDim(left_num);
      block_y = details::GetBlockDim(reduce_num);
      block_dim->x = std::min(block_x, 32);
      block_dim->y =
          std::min(block_y, static_cast<int>(max_num_threads / block_dim->x));
      block_dim->x =
          std::min(block_x, static_cast<int>(max_num_threads / block_dim->y));
      grid_num = details::AlignUp(left_num, block_dim->x);
      reduce_num_per_thread = details::AlignUp(reduce_num, block_dim->y);
    }
    int device_id = paddle::platform::GetCurrentDeviceId();
    int max_mp = paddle::platform::GetGPUMultiProcessors(device_id);
    int max_threads_per_mp =
        paddle::platform::GetGPUMaxThreadsPerMultiProcessor(device_id);
    int max_threads = max_threads_per_mp * max_mp;
    int num_threads = block_dim->x * block_dim->y;
    int max_num_blocks = max_threads / num_threads;

    // set grid size.
    // Whether to set grid.y larger than 1, there are 3 following rules:
    // 1. The number that each thread process should no less than
    //    min_reduce_num_per_threadbut no more than max_reduce_num_per_thread;
    // 2. It should maximize the utilization of SM.
    // So we choose the minimum between input_split_num_1 and input_split_num_3
    // to make each thread process as mush data as possible. Meanwhile,
    // the number cannot be larger than max_reduce_num_per_thread, so we
    // choose the maximum between the result above and input_split_num_2.
    int input_split_num_1 =
        details::AlignUp(reduce_num_per_thread, min_reduce_num_per_thread);
    int input_split_num_2 =
        details::AlignUp(reduce_num_per_thread, max_reduce_num_per_thread);
    int input_split_num_3 = details::AlignUp(max_num_blocks, grid_num);

    grid_dim->x = grid_num;
    grid_dim->y = std::max(std::min(input_split_num_1, input_split_num_3),
                           input_split_num_2);
    // if grid.y > 1, we need launch reduce kernel again.
    if (grid_dim->y > 1) {
      should_reduce_again = true;
    }
  }

  // set block and grid for launch kernel
  // for ReduceHigherDim: if block is enough -> splite reduce_num
  //                     else init block(32, 1) grid(block_num, 1)
  // for others: block(block_num, 1) , grid(left_num, 1)
  void SetBlockDimForHigher(dim3* block_dim, dim3* grid_dim) {
    int last_dim_num = x_dim.back();
    // update left_num
    int grid_z = left_num / last_dim_num;
    left_num = last_dim_num;
    grid_dim->z = grid_z;
    int device_id = paddle::platform::GetCurrentDeviceId();
    int max_mp = paddle::platform::GetGPUMultiProcessors(device_id);
    int max_threads_per_mp =
        paddle::platform::GetGPUMaxThreadsPerMultiProcessor(device_id);
    int max_threads = max_threads_per_mp * max_mp;
    // init
    int num_block = (max_threads / left_num);
    block_dim->x = details::GetBlockDim(left_num);
    grid_dim->x = details::AlignUp(left_num, block_dim->x);
    blocking_size = reduce_num;

    if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
      blocking_size = details::GetLastPow2(reduce_num / num_block);
      if (blocking_size <= 1) {
        blocking_size = details::GetLastPow2(sqrt(reduce_num));
      } else if (blocking_size * 2 < reduce_num) {
        blocking_size *= 2;
      }
      should_reduce_again = true;
      grid_dim->y = details::AlignUp(reduce_num, blocking_size);
    }
  }
#endif

  void SetBlockDim() {
    // init
    should_reduce_again = false;
    dim3 block_dim;
    dim3 grid_dim(left_num, 1, 1);
    blocking_size = reduce_num;

#ifdef PADDLE_WITH_XPU_KP
    if (reduce_last_dim) {
      block_dim.x = 64;
      block_dim.y = reduce_num;
      grid_dim.x = 1;
      grid_dim.y = 8;
    } else {
      block_dim.x = 64;
      block_dim.y = left_num;
      grid_dim.x = 8;
      grid_dim.y = 1;
    }
#else
    if (reduce_type == ReduceType::kReduceHigherDim) {
      SetBlockDimForHigher(&block_dim, &grid_dim);
    } else {
      SetBlockDimForReduceAny(&block_dim, &grid_dim);
    }
#endif

    block = block_dim;
    grid = grid_dim;
  }

 public:
  std::vector<int> reduce_dims_origin;
  std::vector<int> reduce_dim;
  std::vector<int> x_dim;
  std::vector<int> left_dim;
  std::vector<int> x_strides;
  std::vector<int> left_strides;
  std::vector<int> reduce_strides;

  int reduce_type;
  int reduce_num;
  int left_num;
  int blocking_size;
  bool should_reduce_again;
  bool reduce_last_dim;
  Ty* output_data;
  dim3 block;
  dim3 grid;
};

constexpr int kReduceMaxThread = 128;
constexpr int kWarpSize = 32;

__device__ __forceinline__ int SharedMemoryIndex(int index) {
  return (threadIdx.y + index) * blockDim.x + threadIdx.x;
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ T WarpReduce(T val, ReduceOp reducer) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = kWarpSize / 2; stride > 0; stride >>= 1) {
    T temp = paddle::platform::CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  return val;
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockXReduce(T val, ReduceOp reducer) {
  __syncthreads();
  // using details::kWarpSize;
  __shared__ T shared[2 * kWarpSize];
  int block_dim_x = blockDim.x;
  if (blockDim.x > kWarpSize) {
    block_dim_x = blockDim.x / kWarpSize;
    int lane = threadIdx.x % kWarpSize;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int wid = tid / kWarpSize;
    int bid = threadIdx.y;
    val = WarpReduce(val, reducer);
    if (lane == 0) {
      shared[wid] = val;
    }
    __syncthreads();
    val = shared[bid * block_dim_x + lane];
  }

  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = 1; stride < block_dim_x; stride <<= 1) {
    T temp = paddle::platform::CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = val;
  }
  __syncthreads();
  return shared[threadIdx.y];
}

/**
 * @brief BlockYReduce reduce along blockDim.y.
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockYReduce(T val, ReduceOp reducer) {
  __shared__ T shared_memory[1024];
  shared_memory[SharedMemoryIndex(0)] = val;
  for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.y < stride && threadIdx.y + stride < blockDim.y) {
      T temp = shared_memory[SharedMemoryIndex(stride)];
      val = reducer(val, temp);
    }
    shared_memory[SharedMemoryIndex(0)] = val;
  }
  __syncthreads();
  return shared_memory[threadIdx.x];
}


template <typename Tx, 
          typename Ty, 
          typename MPType, 
          typename ReduceOp,
          typename TransformOp>
__global__ void ReduceAnyKernel(const Tx* x, 
                          Ty* y, 
                          ReduceOp reducer,
                          TransformOp transformer, 
                          MPType init, 
                          int reduce_num,
                          int left_num, 
                          bool reduce_lastdim,
                          const IndexCalculator reduce_index_calculator,
                          const IndexCalculator left_index_calculator) {
  int input_idx, left_idx, stride;
  // the last dim gets involved in reduction
  if (reduce_lastdim) {
    input_idx = blockIdx.y * blockDim.x + threadIdx.x;
    left_idx = blockIdx.x * blockDim.y + threadIdx.y;
    stride = gridDim.y * blockDim.x;
  } else {
    input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    left_idx = blockIdx.x * blockDim.x + threadIdx.x;
    stride = gridDim.y * blockDim.y;
  }
  // calculate the offset, means the addr where each thread really start.
  int input_offset = left_index_calculator(left_idx);
  const Tx* input = x + input_offset;
  MPType reduce_var = init;

  // 1. reduce for each thread
  if (left_idx < left_num) {
    // load REDUCE_VEC_SIZE data once, and then compute
    Tx input_reg[REDUCE_VEC_SIZE];
    int bound = reduce_num - (REDUCE_VEC_SIZE - 1) * stride;
    while (input_idx < bound) {
#pragma unroll
      for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
        int reduce_idx = input_idx + i * stride;
        int idx_x = reduce_index_calculator(reduce_idx);
        input_reg[i] = input[idx_x];
      }
#pragma unroll
      for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
        reduce_var =
            reducer(reduce_var, static_cast<MPType>(transformer(input_reg[i])));
      }
      input_idx += REDUCE_VEC_SIZE * stride;
    }

    // deal with the remain part
    int input_idx_tmp = input_idx;
#pragma unroll
    for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
      if (input_idx >= reduce_num) {
        break;
      }
      int reduce_idx = input_idx;
      int idx_x = reduce_index_calculator(reduce_idx);
      input_reg[i] = input[idx_x];
      input_idx += stride;
    }
    input_idx = input_idx_tmp;
#pragma unroll
    for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
      if (input_idx >= reduce_num) {
        break;
      }
      reduce_var =
          reducer(reduce_var, static_cast<MPType>(transformer(input_reg[i])));
      input_idx += stride;
    }
  }

  // 2. reduce in block y
  if (!reduce_lastdim && blockDim.y > 1) {
    reduce_var = BlockYReduce(reduce_var, reducer);
  }
  __syncthreads();

  if (reduce_lastdim) {
    // 3. reduce in block x
    reduce_var = BlockXReduce(reduce_var, reducer);
    if (left_idx < left_num && threadIdx.x == 0) {
      y[blockIdx.y * left_num + left_idx] = static_cast<Ty>(reduce_var);
    }
  } else {
    if (left_idx < left_num && threadIdx.y == 0) {
      y[blockIdx.y * left_num + left_idx] = static_cast<Ty>(reduce_var);
    }
  }
}


template <typename Tx,
          typename Ty,
          typename MPType,
          typename ReduceOp,
          typename TransformOp>
__global__ void ReduceHigherDimKernel(const Tx* x,
                                      Ty* y,
                                      ReduceOp reducer,
                                      TransformOp transformer,
                                      MPType init,
                                      int reduce_num,
                                      int left_num,
                                      int block_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * block_size;

  MPType reduce_var = init;

  if (idx < left_num) {
    int loop = reduce_num - idy;
    loop = loop > block_size ? block_size : loop;

    for (int iy = 0; iy < loop; iy++) {
      int id = (idy + iy) * left_num + idx + blockIdx.z * reduce_num * left_num;
      reduce_var = reducer(reduce_var, static_cast<MPType>(transformer(x[id])));
    }

    y[idx + blockIdx.y * left_num + blockIdx.z * gridDim.y * left_num] =
        static_cast<Ty>(reduce_var);
  }
}

template <typename Tx,
          typename Ty,
          typename MPType,
          typename ReduceOp,
          typename TransformOp>
static void LaunchReduceKernel(const Tx* x_data,
                               Ty* y_data,
                               const ReduceOp& reducer,
                               const TransformOp& transform,
                               MPType init,
                               KPStream stream,
                               ReduceConfig<Ty> config,
                               bool is_mean = false) {
  if (config.reduce_type == kReduceLastDim) {
    int stride_reduce = 1;
    int stride_left = config.reduce_num;
    // for higher performance

    // auto reduce_index_calculator = OneDimIndexCal(stride_reduce);
    // auto left_index_calculator = OneDimIndexCal(stride_left);
    int reduce_rank = config.reduce_strides.size();
    int left_rank = config.left_strides.size();
    auto reduce_index_calculator = IndexCalculator(reduce_rank,
                                                   config.reduce_dim,
                                                   config.reduce_strides,
                                                   config.x_strides);
    auto left_index_calculator = IndexCalculator(
        left_rank, config.left_dim, config.left_strides, config.x_strides);

    kps::DimConfig dim = kps::DimConfig(config.grid.x,
                                        config.grid.y,
                                        config.grid.z,
                                        config.block.x,
                                        config.block.y,
                                        0);
    dim.SetRem(config.reduce_num % config.block.x, 0, 0);

#ifdef PADDLE_WITH_XPU_KP
    auto grid_num = 8;
    auto block_num = 64;
#else
    auto grid_num = config.grid;
    auto block_num = config.block;
#endif
    ReduceAnyKernel<Tx,
                    Ty,
                    MPType,
                    ReduceOp,
                    TransformOp><<<grid_num, block_num, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        init,
        config.reduce_num,
        config.left_num,
        config.reduce_last_dim,
        reduce_index_calculator,
        left_index_calculator);

  } else {
    int reduce_rank = config.reduce_strides.size();
    int left_rank = config.left_strides.size();
    auto reduce_index_calculator = IndexCalculator(reduce_rank,
                                                   config.reduce_dim,
                                                   config.reduce_strides,
                                                   config.x_strides);
    auto left_index_calculator = IndexCalculator(
        left_rank, config.left_dim, config.left_strides, config.x_strides);

    kps::DimConfig dim = kps::DimConfig(config.grid.x,
                                        config.grid.y,
                                        config.grid.z,
                                        config.block.x,
                                        config.block.y,
                                        0);
    dim.SetRem(config.reduce_num % config.block.x, 0, 0);

#ifdef PADDLE_WITH_XPU_KP
    auto grid_num = 8;
    auto block_num = 64;
#else
    auto grid_num = config.grid;
    auto block_num = config.block;
#endif
    ReduceAnyKernel<Tx,
                    Ty,
                    MPType,
                    ReduceOp,
                    TransformOp><<<grid_num, block_num, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        init,
        config.reduce_num,
        config.left_num,
        config.reduce_last_dim,
        reduce_index_calculator,
        left_index_calculator);
  }

  if (config.should_reduce_again) {
    dim3 block;
    dim3 grid;
    if (config.reduce_last_dim) {
      block = dim3(32, 1, 1);
      grid = dim3(details::AlignUp(config.left_num, 32), 1, 1);
    } else {
      block = dim3(config.block.x, 1, 1);
      grid = dim3(config.grid.x, 1, config.grid.z);
    }

    auto last_index = OneDimIndexCal(1);
    auto first_index = OneDimIndexCal(config.left_num);
    kps::DimConfig dim =
        kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
    dim.SetRem(config.left_num % block.x, 0, 0);
#ifdef PADDLE_WITH_XPU_KP
    int grid_size = 8;
    int block_size = 64;
#else
    auto grid_size = grid;
    auto block_size = block;
#endif
    ReduceHigherDimKernel<
        Ty,
        Ty,
        MPType,
        ReduceOp,
        kps::IdentityFunctor<Ty, MPType>><<<grid_size, block_size, 0, stream>>>(
        config.output_data,
        y_data,
        reducer,
        kps::IdentityFunctor<Ty, MPType>(),
        init,
        config.grid.y,
        config.left_num,
        config.grid.y);
  }
}

#if !defined(PADDLE_WITH_XPU_KP)
template <typename Tx,
          typename Ty,
          template <typename> class ReduceOp,
          typename TransformOp>
static typename std::enable_if<!std::is_same<Tx, phi::dtype::float16>::value,
                               void>::type
CubTensorReduceImpl(const Tx* x_data,
                    Ty* y_data,
                    const TransformOp& transform,
                    int reduce_num,
                    const KPDevice& dev_ctx,
                    KPStream stream) {
  auto reducer = ReduceOp<Ty>();
  cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(x_data,
                                                                  transform);
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(nullptr,
                            temp_storage_bytes,
                            trans_x,
                            y_data,
                            reduce_num,
                            reducer,
                            reducer.initial(),
                            stream);
  phi::DenseTensor tmp = phi::Empty<uint8_t, phi::GPUContext>(
      dev_ctx, {static_cast<int64_t>(temp_storage_bytes)});

  auto* temp_storage = dev_ctx.Alloc<uint8_t>(&tmp);

  cub::DeviceReduce::Reduce(temp_storage,
                            temp_storage_bytes,
                            trans_x,
                            y_data,
                            reduce_num,
                            reducer,
                            reducer.initial(),
                            stream);
}

template <typename Tx,
          typename Ty,
          template <typename> class ReduceOp,
          typename TransformOp>
static typename std::enable_if<std::is_same<Tx, phi::dtype::float16>::value,
                               void>::type
CubTensorReduceImpl(const Tx* x_data,
                    Ty* y_data,
                    const TransformOp& transform,
                    int reduce_num,
                    const KPDevice& dev_ctx,
                    KPStream stream) {
  PADDLE_THROW(phi::errors::InvalidArgument(
      "Tx should not be float16 when using cub::DeviceReduce::Reduce()."));
}
#endif  // PADDLE_WITH_XPU_KP

template <typename Tx,
          typename Ty,
          template <typename> class ReduceOp,
          typename TransformOp>
void ReduceKernel(const KPDevice& dev_ctx,
                  const phi::DenseTensor& x,
                  phi::DenseTensor* y,
                  const TransformOp& transform,
                  const std::vector<int>& origin_reduce_dims,
                  bool is_mean = false) {
#ifdef PADDLE_WITH_XPU_KP
  auto stream = dev_ctx.x_context()->xpu_stream;
#else
  auto stream = dev_ctx.stream();
#endif
  dev_ctx.Alloc<Ty>(y);

  auto x_dim = phi::vectorize<int>(x.dims());
  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run(dev_ctx);
  int numel = x.numel();
  // after config.run()
  // SetOutputData for ReduceHigherDim when should_reduce_again is true,
  // temp_output should be stored temp_data in output_data space or stored in
  // y_data;

  phi::DDim tmp_ddim;
  phi::DenseTensor tmp;

  auto x_data = x.data<Tx>();
  auto y_data = y->data<Ty>();

  if (config.reduce_num == 1) {
    std::vector<const DenseTensor*> inputs = {&x};
    std::vector<DenseTensor*> outputs = {y};
    funcs::ElementwiseKernel<Ty>(dev_ctx, inputs, &outputs, transform);
    return;
  }

  config.SetOutputData(y_data, dev_ctx, &tmp);
  constexpr bool kIsTxFP16 = std::is_same<Tx, phi::dtype::float16>::value;
  bool use_cub_reduce = config.reduce_num == numel && !kIsTxFP16;
#ifndef PADDLE_WITH_XPU_KP
  if (use_cub_reduce) {
    if (is_mean) {
      using Div = kps::DivideFunctor<Tx>;
      CubTensorReduceImpl<Tx, Ty, ReduceOp, Div>(x_data,
                                                 y_data,
                                                 Div(config.reduce_num),
                                                 config.reduce_num,
                                                 dev_ctx,
                                                 stream);
    } else {
      CubTensorReduceImpl<Tx, Ty, ReduceOp, TransformOp>(
          x_data, y_data, transform, config.reduce_num, dev_ctx, stream);
    }
    return;
  }
#endif

  using MPType = typename kps::details::MPTypeTrait<Ty>::Type;
  auto reducer = ReduceOp<MPType>();
  // launch ReduceHigherDimKernel
  // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
  // function will be used
  // eg: x_dim = {nz, ny, nx}, nx != 1, axis can be 0 or 1
  //     if axis = 1 then grid.z = nz, grid.y = ny / block_size, grid.x = nx /
  //     32
  //     else grid.z = 1, grid.y = ny / block_size, grid.x = nx /32
  if (config.reduce_type == ReduceType::kReduceHigherDim) {
    kps::DimConfig dim = kps::DimConfig(config.grid.x,
                                        config.grid.y,
                                        config.grid.z,
                                        config.block.x,
                                        config.blocking_size,
                                        0);
    dim.SetRem(config.left_num % config.block.x,
               config.reduce_num % config.blocking_size,
               0);

#ifdef PADDLE_WITH_XPU_KP
    auto grid_num = 8;
    auto block_num = 64;
#else
    auto grid_num = config.grid;
    auto block_num = config.block;
#endif
    ReduceHigherDimKernel<Tx,
                          Ty,
                          MPType,
                          ReduceOp<MPType>,
                          TransformOp><<<grid_num, block_num, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        reducer.initial(),
        config.reduce_num,
        config.left_num,
        config.blocking_size);

    if (config.should_reduce_again) {
      dim3 block = dim3(config.block.x, 1, 1);
      dim3 grid = dim3(config.grid.x, 1, config.grid.z);
      kps::DimConfig dim2 =
          kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
      dim2.SetRem(config.left_num % config.block.x, 0, 0);

#ifdef PADDLE_WITH_XPU_KP
      int grid_size = 8;
      int block_size = 64;
#else
      auto grid_size = grid;
      auto block_size = block;
#endif
      ReduceHigherDimKernel<
          Ty,
          Ty,
          MPType,
          ReduceOp<MPType>,
          kps::IdentityFunctor<Ty,
                               MPType>><<<grid_size, block_size, 0, stream>>>(
          config.output_data,
          y_data,
          reducer,
          kps::IdentityFunctor<Ty, MPType>(config.grid.y),
          reducer.initial(),
          config.grid.y,
          config.left_num,
          config.grid.y);
    }
    return;
  }

  // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used
  LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<MPType>, TransformOp>(
      x_data,
      y_data,
      reducer,
      transform,
      reducer.initial(),
      stream,
      config,
      is_mean);
}

}  // namespace funcs

}  // namespace phi

#endif
