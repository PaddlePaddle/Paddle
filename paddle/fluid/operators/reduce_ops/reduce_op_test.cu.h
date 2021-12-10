// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#ifdef PADDLE_WITH_XPU2
#else
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/fast_divmod.h"
#endif

// Reduce split or not, Whether to use ReduceHigherDim
#define REDUCE_SPLIT_BOUNDARY 512
#define REDUCE_VEC_SIZE 4
#ifdef PADDLE_WITH_XPU2
struct dim3 {
  int x;
  int y;
  int z;
  dim3() {}
  explicit inline dim3(int split_x, int split_y = 1, int split_z = 1) {
    x = split_x;
    y = split_y;
    z = split_z;
  }
};
#endif

namespace paddle {
namespace operators {

namespace kps = paddle::operators::kernel_primitives;

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

// get blockDim for reduceLastDim and reduceAny
static inline int GetBlockDim(int block_dim) {
  return block_dim >= kps::details::kReduceMaxThread
             ? kps::details::kReduceMaxThread
             : GetLastPow2(block_dim);
}

// check reduce rand is valid
static inline void CheckReduceRank(int reduce_rank, int rank) {
  if (rank % 2 == 0) {
    PADDLE_ENFORCE_EQ(reduce_rank, rank / 2,
                      platform::errors::InvalidArgument(
                          "ReduceOp: invalid reduce rank. When rank = %d, "
                          "reduce_rank must be %d, but got %d.",
                          rank, rank / 2, reduce_rank));
  } else {
    auto lower_rank = (rank - 1) / 2;
    auto upper_rank = (rank + 1) / 2;
    PADDLE_ENFORCE_EQ(
        reduce_rank == lower_rank || reduce_rank == upper_rank, true,
        platform::errors::InvalidArgument(
            "ReduceOp: invalid reduce rank. When rank = %d, reduce_rank "
            "must be %d or %d, but got %d.",
            rank, lower_rank, upper_rank, reduce_rank));
  }
}

// convert dims from vector to array
template <typename T, size_t ElementCount, typename VectorLikeType>
static inline paddle::framework::Array<T, ElementCount> VectorToArray(
    const VectorLikeType& vec) {
  PADDLE_ENFORCE_LE(vec.size(), ElementCount,
                    platform::errors::InvalidArgument(
                        "Cub reduce Array: size not match. Received "
                        "vec.size() %d > ElementCount %d.",
                        vec.size(), ElementCount));
  size_t n = static_cast<size_t>(vec.size());
  paddle::framework::Array<T, ElementCount> ret;
  for (size_t i = 0; i < n; ++i) {
    ret[i] = vec[i];
  }
  return ret;
}

}  // namespace details

using Tensor = framework::Tensor;
constexpr int kMaxRank = framework::DDim::kMaxRank;

enum ReduceType {
  kReduceLastDim = 0x01,    // when reduce_dim[0] == x_dim.size() - 1;
  kReduceHigherDim = 0x02,  // ReduceFirstDim or reduceSecondDim
  kReduceAny = 0x03,        // when reduce_dim.size() > 1
};

struct IndexCalculator {
  IndexCalculator(int dim, const std::vector<int>& cal_dims,
                  const std::vector<int>& cal_strides,
                  const std::vector<int>& full_strides)
      : dim(dim) {
    dims = details::VectorToArray<int, kMaxRank>(cal_dims);
    strides = details::VectorToArray<int, kMaxRank>(full_strides);
    reduce_strides = details::VectorToArray<int, kMaxRank>(cal_strides);
    //   std::vector<platform::FastDivMod> cal_divmoders;
    //   // fast divmod
    //   for (auto i : cal_strides) {
    //     cal_divmoders.push_back(platform::FastDivMod(i));
    //   }
    //   divmoders =
    //       details::VectorToArray<platform::FastDivMod,
    //       kMaxRank>(cal_divmoders);
  }

  __device__ inline int operator()(int offset) const {
    int base = offset;
#ifdef PADDLE_WITH_XPU2
    int index = 0;
    //#pragma unroll
    // for (int i = 0; i < kMaxRank; ++i) {
    //   if (i == dim) {
    //     break;
    //   }
    //   index += (offset / reduce_strides[i]) * strides[dims[i]];
    //   offset = offset % reduce_strides[i];
    // }
    return index;
#else
//     int index = 0;
// #pragma unroll
//     for (int i = 0; i < kMaxRank; ++i) {
//       if (i == dim) {
//         break;
//       }
//       auto divmod = divmoders[i].Divmod(offset);
//       index += (divmod.val[0] * strides[dims[i]]);
//       offset = divmod.val[1];
//     }
//     return index;
#endif
  }

  int dim;
  framework::Array<int, kMaxRank> dims;
  framework::Array<int, kMaxRank> strides;
  framework::Array<int, kMaxRank> reduce_strides;
  // framework::Array<platform::FastDivMod, kMaxRank> divmoders;
};

template <bool ReduceLastDim = false>
struct ReduceIndexMapping {
  const kps::DimConfig dim;
  __device__ explicit ReduceIndexMapping(const kps::DimConfig& dims)
      : dim(dims) {}
#ifdef PADDLE_WITH_XPU2
  __device__ int BlockIdX() {
    if (ReduceLastDim) {
      return (cluster_id() / dim.split_num_x % dim.split_num_y);
    } else {
      return cluster_id() % dim.split_num_x;
    }
  }

  __device__ int BlockIdY() {
    if (ReduceLastDim) {
      return (cluster_id() % dim.split_num_x);
    } else {
      return (cluster_id() / dim.split_num_x % dim.split_num_y);
    }
  }

  __device__ int BlockDimX() { return dim.deal_size_x; }

  __device__ int BlockDimY() { return 1; }

  __device__ int GridDimX() {
    if (ReduceLastDim) {
      return dim.split_num_y;
    } else {
      return dim.split_num_x;
    }
  }

  __device__ int GridDimY() {
    if (ReduceLastDim) {
      return dim.split_num_x;
    } else {
      return dim.split_num_y;
    }
  }

  __device__ int GetLoopSize() {
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
  void Run() {
    // step1: update the reduce_dim left_dim and x_dim
    SetReduceDim();

    // step2: get the strides of dim for reduceAny and reduceLastDim
    SetStrides();

    // step3: get the type of reduce
    SetReduceType();

    // step4: set the block and grid for launch kernel
    SetBlockDim();
  }

  // when should_reduce_again is true, we need malloc temp space for temp data
  void SetOutputData(Ty* y_data, const platform::Place& place,
                     framework::Tensor* tmp) {
    if (should_reduce_again) {
      output_data = tmp->mutable_data<Ty>(
          framework::make_ddim(
              {static_cast<int64_t>(left_num * grid.z * grid.y * sizeof(Ty))}),
          place);
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
    //	printf("\nreduce_dim !!!!!\n");
    //	for(int i = 0; i < reduce_dim.size(); i++) {
    //	  printf(" %d ", reduce_dim[i]);
    //	}
    //	printf("\nx_dim !!!!!\n");
    //	for(int i = 0; i < x_dim.size(); i++) {
    //	  printf(" %d ", x_dim[i]);
    //	}
    //	printf("\nleft_dim !!!!!\n");
    //	for(int i = 0; i < left_dim.size(); i++) {
    //	  printf(" %d ", reduce_dim[i]);
    //	}
    //	printf("\nleft_dim end\n");
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
    bool is_last_dim =
        (rank == 2) && (reduce_rank == 1) && (reduce_dim[0] == 1);
    if (rank == reduce_rank || is_last_dim) {
#ifdef PADDLE_WITH_XPU
      reduce_type = static_cast<int>(ReduceType::kReduceAny);
#else
      reduce_type = static_cast<int>(ReduceType::kReduceLastDim);
#endif
    } else if (reduce_rank == 1) {
// ReduceFirstDim and reduceSecondDim
#ifdef PADDLE_WITH_XPU
      if (reduce_dim[0] == 0) {
        reduce_type = static_cast<int>(ReduceType::kReduceHigherDim);
      } else {
        reduce_type = static_cast<int>(ReduceType::kReduceAny);
      }
#else
      reduce_type = static_cast<int>(ReduceType::kReduceHigherDim);
#endif
    } else {
      reduce_type = static_cast<int>(ReduceType::kReduceAny);
    }
  }
#ifndef PADDLE_WITH_XPU2
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
    int device_id = platform::GetCurrentDeviceId();
    int max_mp = platform::GetCUDAMultiProcessors(device_id);
    int max_threads_per_mp =
        platform::GetCUDAMaxThreadsPerMultiProcessor(device_id);
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
    int device_id = platform::GetCurrentDeviceId();
    int max_mp = platform::GetCUDAMultiProcessors(device_id);
    int max_threads_per_mp =
        platform::GetCUDAMaxThreadsPerMultiProcessor(device_id);
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
    int block_num = details::GetBlockDim(reduce_num);
    should_reduce_again = false;
    dim3 block_dim(block_num, 1, 1);
    dim3 grid_dim(left_num, 1, 1);
    blocking_size = reduce_num;
#ifdef PADDLE_WITH_XPU2
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

// when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
// when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
// function will be used
template <typename Tx, typename Ty, typename MPType>
__global__ void ReduceAnyKernel(
    const Tx* x, Ty* y, MPType init, int reduce_num, int left_num,
    bool reduce_last_dim, const int* reduce_strides, const int* left_strides,
    const int* out_strides, const int* reduce_dims, const int* left_dims,
    int reduce_rank, int left_rank, int split_num_x, int split_num_y,
    int split_num_z, int deal_size_x, int deal_size_y, int deal_size_z,
    int rem_x, int rem_y, int rem_z, int div) {
  using Trans = kps::DivideFunctor<Tx>;
  Trans trans_mean = kps::DivideFunctor<Tx>(div);
  int input_idx, left_idx, stride;
  int block_size = 0;
  bool need_store = true;
  int loop_left = 0;
  int tid = 0;
  // the last dim gets involved in reduction
  int store_offset = 0;
  int stride_left = 0;
  auto dim = kps::DimConfig(split_num_x, split_num_y, split_num_z, deal_size_x,
                            deal_size_y, deal_size_z, rem_x, rem_y, rem_z);
  if (reduce_last_dim) {
    auto block = ReduceIndexMapping<true>(dim);
    input_idx = block.BlockIdY() * block.BlockDimX();
    left_idx = block.BlockIdX() * block.BlockDimY() + THREAD_ID_Y;
    stride = block.GridDimY() * block.BlockDimX();
    block_size = block.BlockDimX();
    need_store = (THREAD_ID_X == 0) && (left_idx < left_num);
    store_offset = block.BlockIdY() * left_num + left_idx;
    loop_left =
        left_num - left_idx;  // min(block.GetLoopSize(), left_num - left_idx);
    stride_left = 1;
    tid = THREAD_ID_X;
  } else {
    auto block = ReduceIndexMapping<false>(dim);
    input_idx = block.BlockIdY() * block.BlockDimY();
    left_idx = block.BlockIdX() * block.BlockDimX() + THREAD_ID_X;
    stride = block.GridDimY() * block.BlockDimY();
    block_size = block.BlockDimY();
    need_store = (THREAD_ID_Y == 0) && (left_idx < left_num);
    loop_left =
        left_num - left_idx;  // min(block.GetLoopSize(), left_num - left_idx);
    stride_left = block.BlockDimX() * block.GridDimX();
    store_offset = block.BlockIdY() * left_num + left_idx;
    tid = THREAD_ID_Y;
  }
  // calculate the offset, means the addr where each thread really start.
  // 1. reduce for each thread
  MPType input_compute[REDUCE_VEC_SIZE];
  Tx input_reg[REDUCE_VEC_SIZE];
  int input_idx_tmp = input_idx;
  for (int i = 0; i < loop_left; i += stride_left) {
    int input_offset = kps::IndexCalculator(left_strides, out_strides,
                                            left_dims, left_rank, left_idx + i);
    // const Tx _global_ptr * input = x + input_offset;
    MPType reduce_var = init;
    // load REDUCE_VEC_SIZE data once, and then compute
    int bound = reduce_num - (REDUCE_VEC_SIZE - 1) * stride;
    input_idx = input_idx_tmp;
    for (; input_idx + block_size < bound;
         input_idx += REDUCE_VEC_SIZE * stride) {
      // printf("befor ReadDataReduce_0 \n");
      kps::ReadDataReduce<Tx, Tx, 1, REDUCE_VEC_SIZE, 1, 1,
                          kps::IdentityFunctor<Tx>, false>(
          &input_reg[0], x + input_offset, input_idx, reduce_strides,
          out_strides, reduce_dims, reduce_rank, 1, reduce_num, 1, stride,
          kps::IdentityFunctor<Tx>(), reduce_last_dim);
      // printf("befor Unary_0 \n");
      kps::ElementwiseUnary<Tx, MPType, REDUCE_VEC_SIZE, 1, 1, Trans>(
          &input_compute[0], &input_reg[0], trans_mean);
      // printf("befor Reduce_0 \n");
      kps::Reduce<MPType, REDUCE_VEC_SIZE, 1, 1, kps::AddFunctor<MPType>,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &input_compute[0], kps::AddFunctor<MPType>(),
          reduce_last_dim);
    }

    kps::Init<MPType, REDUCE_VEC_SIZE>(&input_compute[0], init);
    // printf("befor ReadDataReduce_1 \n");
    kps::ReadDataReduce<Tx, MPType, 1, REDUCE_VEC_SIZE, 1, 1, Trans, true>(
        &input_compute[0], x + input_offset, input_idx, reduce_strides,
        out_strides, reduce_dims, reduce_rank, 1, reduce_num - input_idx, 1,
        stride, trans_mean, reduce_last_dim);
    // printf("befor Reduce_1 \n");
    kps::Reduce<MPType, REDUCE_VEC_SIZE, 1, 1, kps::AddFunctor<MPType>,
                kps::details::ReduceMode::kLocalMode>(
        &reduce_var, &input_compute[0], kps::AddFunctor<MPType>(),
        reduce_last_dim);

    // printf("befor Reduce_2 loop %d idx %d bound %d input_idx %d reduce_num %f
    // input_offset %d\n", loop_left, i, bound, input_idx, reduce_var,
    // input_offset);
    kps::Reduce<MPType, 1, 1, 1, kps::AddFunctor<MPType>,
                kps::details::kGlobalMode>(
        &reduce_var, &reduce_var, kps::AddFunctor<MPType>(), reduce_last_dim);
    if (need_store) {
      // printf("befor Write_data \n");
      Ty result = static_cast<Ty>(reduce_var);
      LM2GM(&result, y + store_offset + i, sizeof(Ty));
    }
  }
}

template <typename Tx, typename Ty, typename MPType, typename ReduceOp,
          typename TransformOp>
__global__ void ReduceHigherDimKernel(
    const Tx* x, Ty* y, ReduceOp reducer, TransformOp transformer, MPType init,
    int reduce_num, int left_num, int blocking_size, int split_num_x,
    int split_num_y, int split_num_z, int deal_size_x, int deal_size_y,
    int deal_size_z, int rem_x, int rem_y, int rem_z, int div) {
  using Trans = kps::DivideFunctor<Tx>;
  Trans trans_mean = kps::DivideFunctor<Tx>(div);
  auto dim = kps::DimConfig(split_num_x, split_num_y, split_num_z, deal_size_x,
                            deal_size_y, deal_size_z, rem_x, rem_y, rem_z);
  // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
  // function will be used
  auto block = ReduceIndexMapping<false>(dim);
  int idy = block.BlockIdY() * blocking_size;
  int idx = block.BlockIdX() * block.BlockDimX();
  int idz = BLOCK_ID_Z * left_num;
  int stride = dim.split_num_x * dim.deal_size_x;
  int size = left_num - dim.rem_x;
  int loop_size = min(reduce_num - idy, blocking_size);
  int store_offset = block.BlockIdY() * left_num + idz * block.GridDimY();
  int block_offset = idy * left_num + idz * reduce_num;
  // const Tx* input = (x + block_offset);
  Tx reduce_input;
  for (; idx < size; idx += stride) {
    MPType reduce_var = init;
    MPType reduce_compute = init;
    for (int loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
      // printf("befor ReadData_0 %d !!!\n", (block_offset + loop_idx * left_num
      // + idx));
      kps::ReadData<Tx, Tx, 1, 1, 1, false>(
          &reduce_input, x + block_offset + loop_idx * left_num + idx,
          block.BlockDimX(), 1, 1, left_num);
      // printf("after ReadData_0 !!!");
      kps::ElementwiseUnary<Tx, MPType, REDUCE_VEC_SIZE, 1, 1, Trans>(
          &reduce_compute, &reduce_input, trans_mean);
      // printf("after Reduce_0 !!!");
      kps::Reduce<MPType, 1, 1, 1, kps::AddFunctor<MPType>,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &reduce_compute, kps::AddFunctor<MPType>(), false);
    }
    Ty result = static_cast<Ty>(reduce_var);
    // printf("befor WriteData_0 !!! %d \n", store_offset + idx);
    kps::WriteData<Ty, 1, 1, 1, false>(y + store_offset + idx, &result,
                                       block.BlockDimX());
    // printf("after WriteData_0 !!! %d \n", store_offset + idx);
  }

  if (idx < left_num) {
    MPType reduce_var = init;
    MPType reduce_compute = init;
    for (int loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
      //  printf("befor ReadData_1 !!!");
      kps::ReadData<Tx, Tx, 1, 1, 1, true>(
          &reduce_input, x + block_offset + loop_idx * left_num + idx,
          dim.rem_x, 1, 1, left_num);
      // printf("after ReadData_0 !!!");
      kps::ElementwiseUnary<Tx, MPType, REDUCE_VEC_SIZE, 1, 1, Trans>(
          &reduce_compute, &reduce_input, trans_mean);
      //  printf("after Unary_1 !!!");
      kps::Reduce<MPType, 1, 1, 1, kps::AddFunctor<MPType>,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &reduce_compute, kps::AddFunctor<MPType>(), false);
      //  printf("after Reduce !!!");
    }
    Ty result = static_cast<Ty>(reduce_var);
    // printf("befor WriteData_1 !!!");
    kps::WriteData<Ty, 1, 1, 1, true>(y + store_offset + idx, &result,
                                      dim.rem_x);
  }
}

template <typename Tx, typename Ty, typename MPType, typename ReduceOp>
static void LaunchReduceKernel(const Tx* x_data, Ty* y_data,
                               const ReduceOp& reducer, MPType init,
                               XPUStream stream, ReduceConfig<Ty> config) {
  using TransformOp = typename ReduceOp::Transformer;

//  if (config.reduce_type == kReduceLastDim) {
//    int stride_reduce = 1;
//    int stride_left = config.reduce_num;
//    // for higher performance
//    auto reduce_index_calculator = OneDimIndexCal(stride_reduce);
//    auto left_index_calculator = OneDimIndexCal(stride_left);
//
//    kps::DimConfig dim =
//        kps::DimConfig(config.grid.x, config.grid.y, config.grid.z,
//                       config.block.x, config.block.y, 0);
//    dim.SetRem(config.reduce_num % config.block.x, 0, 0);
//	int split_num_x = dim.split_num_x;
//	int split_num_y = dim.split_num_y;
//	int split_num_z = dim.split_num_z;
//	int deal_size_x = dim.deal_size_x;
//	int deal_size_y = dim.deal_size_y;
//	int deal_size_z = dim.deal_size_z;
//	int rem_x = dim.rem_x;
//	int rem_y = dim.rem_y;
//	int rem_z = dim.rem_z;
//
//#ifdef PADDLE_WITH_XPU2
//    ReduceAnyKernel<Tx, Ty, MPType, ReduceOp, TransformOp,
//                    OneDimIndexCal><<<8, 64, 0, stream>>>(
//        x_data, config.output_data, reducer, TransformOp(config.reduce_num),
//        init, config.reduce_num, config.left_num, config.reduce_last_dim,
//        reduce_index_calculator, left_index_calculator, dim);
//#else
//    ReduceAnyKernel<Tx, Ty, MPType, ReduceOp, TransformOp,
//                    OneDimIndexCal><<<config.grid, config.block, 0, stream>>>(
//        x_data, config.output_data, reducer, TransformOp(config.reduce_num),
//        init, config.reduce_num, config.left_num, config.reduce_last_dim,
//        reduce_index_calculator, left_index_calculator, dim);
//#endif
//
//  } else {
//  }
#ifdef PADDLE_WITH_XPU2
  int reduce_rank = config.reduce_strides.size();
  int left_rank = config.left_strides.size();
  auto reduce_index_calculator = IndexCalculator(
      reduce_rank, config.reduce_dim, config.reduce_strides, config.x_strides);
  auto left_index_calculator = IndexCalculator(
      left_rank, config.left_dim, config.left_strides, config.x_strides);

  kps::DimConfig dim =
      kps::DimConfig(config.grid.x, config.grid.y, config.grid.z,
                     config.block.x, config.block.y, 0);
  dim.SetRem(config.reduce_num % config.block.x, 0, 0);
  int split_num_x = dim.split_num_x;
  int split_num_y = dim.split_num_y;
  int split_num_z = dim.split_num_z;
  int deal_size_x = dim.deal_size_x;
  int deal_size_y = dim.deal_size_y;
  int deal_size_z = dim.deal_size_z;
  int rem_x = dim.rem_x;
  int rem_y = dim.rem_y;
  int rem_z = dim.rem_z;
  // printf(
  //     "Reduce Any : this reduceAny split_nx %d, split_ny %d, splist_nz %d
  //     deal_size_x %d "
  //     "deal_size_y %d deal_size_z %d reduce_rank %d left_rank %d\n",
  //     config.grid.x, config.grid.y, config.grid.z, config.block.x,
  //     config.block.y, config.reduce_num % config.block.x, reduce_rank,
  //     left_rank);
  int* reduce_strides = nullptr;
  int* left_strides = nullptr;
  int* output_strides = nullptr;
  int* reduce_dims = nullptr;
  int* left_dims = nullptr;
  // reduce_rank, left_rank
  int strides_reduce_array[kMaxRank];
  int strides_left_array[kMaxRank];
  int strides_out_array[kMaxRank];
  int dim_reduce_array[kMaxRank];
  int dim_left_array[kMaxRank];
  // printf("memcpy stride_reduce_array\n");
  memcpy(strides_reduce_array, config.reduce_strides.data(),
         reduce_rank * sizeof(int));
  // printf("memcpy stride_left_array\n");
  memcpy(strides_left_array, config.left_strides.data(),
         left_rank * sizeof(int));
  ;
  // printf("memcpy stride_out_array\n");
  memcpy(strides_out_array, config.x_strides.data(),
         config.x_strides.size() * sizeof(int));
  // printf("memcpy dim_reduce_array\n");
  memcpy(dim_reduce_array, config.reduce_dim.data(), reduce_rank * sizeof(int));
  // printf("memcpy dim_left_array\n");
  memcpy(dim_left_array, config.left_dim.data(), left_rank * sizeof(int));
  // malloc
  int ret = xpu_malloc((void**)&reduce_strides,
                       framework::DDim::kMaxRank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void**)&left_strides,
                   framework::DDim::kMaxRank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void**)&output_strides,
                   framework::DDim::kMaxRank * sizeof(int));
  assert(ret == 0);
  ret =
      xpu_malloc((void**)&reduce_dims, framework::DDim::kMaxRank * sizeof(int));
  assert(ret == 0);
  ret = xpu_malloc((void**)&left_dims, framework::DDim::kMaxRank * sizeof(int));
  assert(ret == 0);
  // memcpy
  // printf("HOST_TO_DEVICE reduce_strides\n");
  if (reduce_rank > 0)
    ret = xpu_memcpy(reduce_strides, strides_reduce_array,
                     reduce_rank * sizeof(int), XPU_HOST_TO_DEVICE);
  assert(ret == 0);
  // printf("HOST_TO_DEVICE left_strides\n");
  if (left_rank > 0)
    ret = xpu_memcpy(left_strides, strides_left_array, left_rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
  assert(ret == 0);
  // printf("HOST_TO_DEVICE output_strides\n");
  if (config.x_strides.size() > 0)
    ret = xpu_memcpy(output_strides, strides_out_array,
                     config.x_strides.size() * sizeof(int), XPU_HOST_TO_DEVICE);
  assert(ret == 0);
  // printf("HOST_TO_DEVICE reduce_dims\n");
  if (reduce_rank > 0)
    ret = xpu_memcpy(reduce_dims, dim_reduce_array, reduce_rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
  assert(ret == 0);
  // printf("HOST_TO_DEVICE  left_dims\n");
  if (left_rank > 0)
    ret = xpu_memcpy(left_dims, dim_left_array, left_rank * sizeof(int),
                     XPU_HOST_TO_DEVICE);
  assert(ret == 0);
  // printf("before ReduceAnyKernel \n");
  int div = config.reduce_num;
  ReduceAnyKernel<Tx, Ty, MPType><<<8, 64, stream>>>(
      x_data, config.output_data, init, config.reduce_num, config.left_num,
      config.reduce_last_dim, reduce_strides, left_strides, output_strides,
      reduce_dims, left_dims, reduce_rank, left_rank, split_num_x, split_num_y,
      split_num_z, deal_size_x, deal_size_y, deal_size_z, rem_x, rem_y, rem_z,
      div);

  // printf("after ReduceAnyKernel \n");
  xpu_free(reduce_strides);
  xpu_free(left_strides);
  xpu_free(output_strides);
  xpu_free(reduce_dims);
  xpu_free(left_dims);
// printf("after free \n");
#else
//    ReduceAnyKernel<Tx, Ty, MPType, ReduceOp, TransformOp,
//                    IndexCalculator><<<config.grid, config.block, 0,
//                    stream>>>(
//        x_data, config.output_data, reducer, TransformOp(config.reduce_num),
//        init, config.reduce_num, config.left_num, config.reduce_last_dim,
//        reduce_index_calculator, left_index_calculator, dim);
#endif

  if (config.should_reduce_again) {
    // printf("Reduce again \n");
    dim3 block;
    dim3 grid;
    if (config.reduce_last_dim) {
      block = dim3(32, 1, 1);
      grid = dim3(details::AlignUp(config.left_num, 32), 1, 1);
    } else {
      block = dim3(config.block.x, 1, 1);
      grid = dim3(config.grid.x, 1, config.grid.z);
    }
    int div = 1;
    auto last_index = OneDimIndexCal(1);
    auto first_index = OneDimIndexCal(config.left_num);
    kps::DimConfig dim2 =
        kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
    dim2.SetRem(config.left_num % block.x, 0, 0);
    int h_split_num_x = dim2.split_num_x;
    int h_split_num_y = dim2.split_num_y;
    int h_split_num_z = dim2.split_num_z;
    int h_deal_size_x = dim2.deal_size_x;
    int h_deal_size_y = dim2.deal_size_y;
    int h_deal_size_z = dim2.deal_size_z;
    int h_rem_x = dim2.rem_x;
    int h_rem_y = dim2.rem_y;
    int h_rem_z = dim2.rem_z;
#ifdef PADDLE_WITH_XPU2
    // printf(
    //     "Reduce Higher : this reduceAny split_nx %d, split_ny %d, splist_nz
    //     %d deal_size_x %d "
    //     "deal_size_y %d deal_size_z %d \n", h_split_num_x, h_split_num_y,
    //     h_split_num_z, h_deal_size_x, h_deal_size_y, h_deal_size_z);
    ReduceHigherDimKernel<Ty, Ty, MPType, ReduceOp,
                          kps::IdentityFunctor<Ty, MPType>><<<8, 64, stream>>>(
        config.output_data, y_data, reducer,
        kps::IdentityFunctor<Ty, MPType>(config.grid.y), init, config.grid.y,
        config.left_num, config.grid.y, h_split_num_x, h_split_num_y,
        h_split_num_z, h_deal_size_x, h_deal_size_y, h_deal_size_z, h_rem_x,
        h_rem_y, h_rem_z, div);
#else
    ReduceHigherDimKernel<
        Ty, Ty, MPType, ReduceOp,
        kps::IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
        config.output_data, y_data, reducer,
        kps::IdentityFunctor<Ty, MPType>(config.grid.y), init, config.grid.y,
        config.left_num, config.grid.y, dim);
#endif
  }
}

template <typename Tx, typename Ty,
          template <typename, typename> class ReduceOp>
void TensorReduceFunctorImpl(const framework::Tensor& x, framework::Tensor* y,
                             std::vector<int> origin_reduce_dims,
                             XPUStream stream) {
  auto x_dim = framework::vectorize<int>(x.dims());
  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();
  int numel = x.numel();
  // after config.run()
  // SetOutputData for ReduceHigherDim when should_reduce_again is true,
  // temp_output should be stored temp_data in output_data space or stored in
  // y_data;
  framework::Tensor tmp;
  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>(x.place());
  // attention
  if (config.reduce_num == 1) {
    auto out_dims = y->dims();
    if (x.type() == y->type()) {
      framework::TensorCopy(x, y->place(), y);
      y->Resize(out_dims);
    } else {
#ifndef PADDLE_WITH_XPU2
      auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
          paddle::platform::DeviceContextPool::Instance().Get(x.place()));
      framework::VisitDataType(
          static_cast<framework::proto::VarType::Type>(y->type()),
          CastOpFunctor<platform::CUDADeviceContext, Tx>(&x, y, *dev_ctx));
#endif
    }
    return;
  }
  config.SetOutputData(y_data, x.place(), &tmp);
  bool use_cub_reduce = (config.reduce_num == numel) &&
                        (!std::is_same<Tx, paddle::platform::float16>::value);
#ifndef PADDLE_WITH_XPU2
  if (use_cub_reduce) {
    // launch CUB::Reduce
    using TransformOp = typename ReduceOp<Tx, Ty>::Transformer;
    auto reducer = ReduceOp<Tx, Ty>();
    cub::TransformInputIterator<Ty, TransformOp, const Tx*> trans_x(
        x_data, TransformOp(config.reduce_num));
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, trans_x, y_data,
                              config.reduce_num, reducer, reducer.initial(),
                              stream);
    framework::Tensor tmp;
    auto* temp_storage = tmp.mutable_data<uint8_t>(
        framework::make_ddim({static_cast<int64_t>(temp_storage_bytes)}),
        x.place());
    cub::DeviceReduce::Reduce(temp_storage, temp_storage_bytes, trans_x, y_data,
                              config.reduce_num, reducer, reducer.initial(),
                              stream);

    return;
  }
#endif
  using MPType = typename details::MPTypeTrait<Ty>::Type;
  auto reducer = ReduceOp<Tx, MPType>();
  // launch ReduceHigherDimKernel
  // when reduce_dim.size() == 1 and reduce_dim[0] != x_dim.size() - 1, this
  // function will be used
  // eg: x_dim = {nz, ny, nx}, nx != 1, axis can be 0 or 1
  //     if axis = 1 then grid.z = nz, grid.y = ny / block_size, grid.x = nx /
  //     32
  //     else grid.z = 1, grid.y = ny / block_size, grid.x = nx /32
  if (config.reduce_type == ReduceType::kReduceHigherDim) {
    using TransformOp = typename ReduceOp<Tx, MPType>::Transformer;
    kps::DimConfig dim =
        kps::DimConfig(config.grid.x, config.grid.y, config.grid.z,
                       config.block.x, config.blocking_size, 0);
    dim.SetRem(config.left_num % config.block.x,
               config.reduce_num % config.blocking_size, 0);
    int split_num_x = dim.split_num_x;
    int split_num_y = dim.split_num_y;
    int split_num_z = dim.split_num_z;
    int deal_size_x = dim.deal_size_x;
    int deal_size_y = dim.deal_size_y;
    int deal_size_z = dim.deal_size_z;
    int rem_x = dim.rem_x;
    int rem_y = dim.rem_y;
    int rem_z = dim.rem_z;
// printf(
//     "Reduce Higher :  split_nx %d, split_ny %d, splist_nz %d deal_size_x %d "
//     "deal_size_y %d deal_size_z %d \n", split_num_x, split_num_y,
//     split_num_z,deal_size_x,deal_size_y,deal_size_z);

#ifdef PADDLE_WITH_XPU2
    int div = config.reduce_num;
    ReduceHigherDimKernel<Tx, Ty, MPType, ReduceOp<Tx, MPType>,
                          TransformOp><<<8, 64, stream>>>(
        x_data, config.output_data, reducer, TransformOp(config.reduce_num),
        reducer.initial(), config.reduce_num, config.left_num,
        config.blocking_size, split_num_x, split_num_y, split_num_z,
        deal_size_x, deal_size_y, deal_size_z, rem_x, rem_y, rem_z, div);
#else
//  ReduceHigherDimKernel<
//      Tx, Ty, MPType, ReduceOp<Tx, MPType>,
//      TransformOp><<<config.grid, config.block, 0, stream>>>(
//      x_data, config.output_data, reducer, TransformOp(config.reduce_num),
//      reducer.initial(), config.reduce_num, config.left_num,
//      config.blocking_size, dim);
#endif

    if (config.should_reduce_again) {
      dim3 block = dim3(config.block.x, 1, 1);
      dim3 grid = dim3(config.grid.x, 1, config.grid.z);
      kps::DimConfig dim2 =
          kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
      dim2.SetRem(config.left_num % config.block.x, 0, 0);
      split_num_x = dim2.split_num_x;
      split_num_y = dim2.split_num_y;
      split_num_z = dim2.split_num_z;
      deal_size_x = dim2.deal_size_x;
      deal_size_y = dim2.deal_size_y;
      deal_size_z = dim2.deal_size_z;
      rem_x = dim2.rem_x;
      rem_y = dim2.rem_y;
      rem_z = dim2.rem_z;

#ifdef PADDLE_WITH_XPU2
      int div = 1;
      ReduceHigherDimKernel<
          Ty, Ty, MPType, ReduceOp<Tx, MPType>,
          kps::IdentityFunctor<Ty, MPType>><<<8, 64, stream>>>(
          config.output_data, y_data, reducer,
          kps::IdentityFunctor<Ty, MPType>(config.grid.y), reducer.initial(),
          config.grid.y, config.left_num, config.grid.y, split_num_x,
          split_num_y, split_num_z, deal_size_x, deal_size_y, deal_size_z,
          rem_x, rem_y, rem_z, div);
#else
//    ReduceHigherDimKernel<
//        Ty, Ty, MPType, ReduceOp<Tx, MPType>,
//        kps::IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
//        config.output_data, y_data, reducer,
//        kps::IdentityFunctor<Ty, MPType>(config.grid.y), reducer.initial(),
//        config.grid.y, config.left_num, config.grid.y, dim2);
#endif
    }
    return;
  }

  // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used
  LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<Tx, MPType>>(
      x_data, y_data, reducer, reducer.initial(), stream, config);
}

// template <typename Tx, template <typename, typename> class ReduceOp>
// struct TensorReduceFunc {
//  const framework::Tensor& x;
//  framework::Tensor* y;
//  std::vector<int> origin_reduce_dims;
//  gpuStream_t stream;
//  TensorReduceFunc(const framework::Tensor& x, framework::Tensor* y,
//                   std::vector<int> origin_reduce_dims, gpuStream_t stream)
//      : x(x), y(y), origin_reduce_dims(origin_reduce_dims), stream(stream) {}
//
//  template <typename Ty>
//  void apply() const {
//    TensorReduceFunctorImpl<Tx, Ty, ReduceOp>(x, y, origin_reduce_dims,
//    stream);
//  }
//};

}  // namespace operators
}  // namespace paddle
