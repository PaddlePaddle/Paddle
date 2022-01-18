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

// CUDA and HIP use same api
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

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
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/fluid/string/string_helper.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/elementwise_base.h"

// Reduce split or not, Whether to use ReduceHigherDim
#define REDUCE_SPLIT_BOUNDARY 512
#define REDUCE_VEC_SIZE 4

namespace kps = paddle::operators::kernel_primitives;

namespace pten {
namespace kernels {

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
    PADDLE_ENFORCE_EQ(reduce_rank,
                      rank / 2,
                      paddle::platform::errors::InvalidArgument(
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
        paddle::platform::errors::InvalidArgument(
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
static inline paddle::framework::Array<T, ElementCount> VectorToArray(
    const VectorLikeType& vec) {
  PADDLE_ENFORCE_LE(vec.size(),
                    ElementCount,
                    paddle::platform::errors::InvalidArgument(
                        "Cub reduce Array: size not match. Received "
                        "vec.size() %d > ElementCount %d.",
                        vec.size(),
                        ElementCount));
  size_t n = static_cast<size_t>(vec.size());
  paddle::framework::Array<T, ElementCount> ret;
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
                        paddle::platform::errors::InvalidArgument(
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

constexpr int kMaxRank = paddle::framework::DDim::kMaxRank;

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
    std::vector<paddle::platform::FastDivMod> cal_divmoders;
    // fast divmod
    for (auto i : cal_strides) {
      cal_divmoders.push_back(paddle::platform::FastDivMod(i));
    }
    divmoders = details::VectorToArray<paddle::platform::FastDivMod, kMaxRank>(
        cal_divmoders);
  }

  __device__ inline int operator()(int offset) const {
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
  }

  int dim;
  paddle::framework::Array<int, kMaxRank> dims;
  paddle::framework::Array<int, kMaxRank> strides;
  paddle::framework::Array<paddle::platform::FastDivMod, kMaxRank> divmoders;
};

template <bool ReduceLastDim = false>
struct ReduceIndexMapping {
  const kps::DimConfig dim;
  HOSTDEVICE explicit ReduceIndexMapping(const kps::DimConfig& dims)
      : dim(dims) {}

  __device__ __forceinline__ int BlockIdX() {
#ifdef PADDLE_WITH_XPU2
    if (ReduceLastDim) {
      return (cluster_id() / dim.split_num_x % dim.split_num_y);
    } else {
      return cluster_id() % dim.split_num_x;
    }
#else
    return blockIdx.x;
#endif
  }

  __device__ __forceinline__ int BlockIdY() {
#ifdef PADDLE_WITH_XPU2
    if (ReduceLastDim) {
      return (cluster_id() % dim.split_num_x);
    } else {
      return (cluster_id() / dim.split_num_x % dim.split_num_y);
    }
#else
    return blockIdx.y;
#endif
  }

  __device__ __forceinline__ int BlockDimX() {
#ifdef PADDLE_WITH_XPU2
    return dim.deal_size_x;
#else
    return blockDim.x;
#endif
  }

  __device__ __forceinline__ int BlockDimY() {
#ifdef PADDLE_WITH_XPU2
    return dim.deal_size_y;
#else
    return blockDim.y;
#endif
  }

  __device__ __forceinline__ int GridDimX() {
#ifdef PADDLE_WITH_XPU2
    if (ReduceLastDim) {
      return dim.split_num_y;
    } else {
      return dim.split_num_x;
    }
#else
    return gridDim.x;
#endif
  }

  __device__ __forceinline__ int GridDimY() {
#ifdef PADDLE_WITH_XPU2
    if (ReduceLastDim) {
      return dim.split_num_x;
    } else {
      return dim.split_num_y;
    }
#else
    return gridDim.y;
#endif
  }

  __device__ __forceinline__ int GetLoopSize() {
#ifdef PADDLE_WITH_XPU2
    if (ReduceLastDim) {
      return dim.deal_size_y;
    } else {
      return dim.deal_size_x;
    }
#else
    return 1;
#endif
  }
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
  void SetOutputData(Ty* y_data,
                     const paddle::platform::Place& place,
                     pten::DenseTensor* tmp) {
    if (should_reduce_again) {
      tmp->ResizeAndAllocate(paddle::framework::make_ddim(
          {static_cast<int64_t>(left_num * grid.z * grid.y * sizeof(Ty))}));
      output_data = tmp->mutable_data<Ty>();
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
    bool is_last_dim =
        (rank == 2) && (reduce_rank == 1) && (reduce_dim[0] == 1);
    if (rank == reduce_rank || is_last_dim) {
      reduce_type = static_cast<int>(ReduceType::kReduceLastDim);
    } else if (reduce_rank == 1) {
// ReduceFirstDim and reduceSecondDim
#ifdef PADDLE_WITH_XPU2
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

  void SetBlockDim() {
    // init
    int block_num = details::GetBlockDim(reduce_num);
    should_reduce_again = false;
    dim3 block_dim(block_num, 1, 1);
    dim3 grid_dim(left_num, 1, 1);
    blocking_size = reduce_num;
#ifdef PADDLE_WITH_XPU2
    if (reduce_last_dim) {
      block_dim.x = 128;
      block_dim.y = reduce_num;
      grid_dim.x = 8;
      grid_dim.y = 1;
    } else {
      block_dim.x = 128;
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
template <typename Tx,
          typename Ty,
          typename MPType,
          typename ReduceOp,
          typename TransformOp,
          typename Calculator>
__global__ void ReduceAnyKernel(const Tx* x,
                                Ty* y,
                                ReduceOp reducer,
                                TransformOp transformer,
                                MPType init,
                                int reduce_num,
                                int left_num,
                                bool reduce_last_dim,
                                const Calculator reduce_index_calculator,
                                const Calculator left_index_calculator,
                                const kps::DimConfig dim) {
  int input_idx, left_idx, stride;
  int block_size = 0;
  bool need_store = true;
  int loop_left = 0;
  int tid = 0;
  // the last dim gets involved in reduction
  int store_offset = 0;
  int stride_left = 0;
  if (reduce_last_dim) {
    auto block = ReduceIndexMapping<true>(dim);
    input_idx = block.BlockIdY() * block.BlockDimX();
    left_idx = block.BlockIdX() * block.BlockDimY() + THREAD_ID_Y;
    stride = block.GridDimY() * block.BlockDimX();
    block_size = block.BlockDimX();
    need_store = (THREAD_ID_X == 0) && (left_idx < left_num);
    store_offset = block.BlockIdY() * left_num + left_idx;
    loop_left = min(block.GetLoopSize(), left_num - left_idx);
    stride_left = 1;
    tid = threadIdx.x;
  } else {
    auto block = ReduceIndexMapping<false>(dim);
    input_idx = block.BlockIdY() * block.BlockDimY();
    left_idx = block.BlockIdX() * block.BlockDimX() + THREAD_ID_X;
    stride = block.GridDimY() * block.BlockDimY();
    block_size = block.BlockDimY();
    need_store = (THREAD_ID_Y == 0) && (left_idx < left_num);
    loop_left = min(block.GetLoopSize(), left_num - left_idx);
    stride_left = block.BlockDimX() * block.GridDimX();
    store_offset = block.BlockIdY() * left_num + left_idx;
    tid = threadIdx.y;
  }
  // calculate the offset, means the addr where each thread really start.
  // 1. reduce for each thread
  MPType input_compute[REDUCE_VEC_SIZE];
  Tx input_reg[REDUCE_VEC_SIZE];
  for (int i = 0; i < loop_left; i += stride_left) {
    int input_offset = left_index_calculator(left_idx + i);
    const Tx* input = x + input_offset;
    MPType reduce_var = init;
    // load REDUCE_VEC_SIZE data once, and then compute
    int bound = reduce_num - (REDUCE_VEC_SIZE - 1) * stride;
    for (; input_idx + block_size < bound;
         input_idx += REDUCE_VEC_SIZE * stride) {
      kps::ReadDataReduce<Tx,
                          Tx,
                          1,
                          REDUCE_VEC_SIZE,
                          1,
                          1,
                          Calculator,
                          kps::IdentityFunctor<Tx>,
                          false>(&input_reg[0],
                                 input,
                                 input_idx,
                                 reduce_index_calculator,
                                 1,
                                 reduce_num,
                                 1,
                                 stride,
                                 kps::IdentityFunctor<Tx>(),
                                 reduce_last_dim);
      kps::ElementwiseUnary<Tx, MPType, REDUCE_VEC_SIZE, 1, 1, TransformOp>(
          &input_compute[0], &input_reg[0], transformer);
      kps::Reduce<MPType,
                  REDUCE_VEC_SIZE,
                  1,
                  1,
                  ReduceOp,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &input_compute[0], reducer, reduce_last_dim);
    }

    kps::Init<MPType, REDUCE_VEC_SIZE>(&input_compute[0], init);
    kps::ReadDataReduce<Tx,
                        MPType,
                        1,
                        REDUCE_VEC_SIZE,
                        1,
                        1,
                        Calculator,
                        TransformOp,
                        true>(&input_compute[0],
                              input,
                              input_idx,
                              reduce_index_calculator,
                              1,
                              reduce_num - input_idx,
                              1,
                              stride,
                              transformer,
                              reduce_last_dim);
    kps::Reduce<MPType,
                REDUCE_VEC_SIZE,
                1,
                1,
                ReduceOp,
                kps::details::ReduceMode::kLocalMode>(
        &reduce_var, &input_compute[0], reducer, reduce_last_dim);

    kps::Reduce<MPType, 1, 1, 1, ReduceOp, kps::details::kGlobalMode>(
        &reduce_var, &reduce_var, reducer, reduce_last_dim);
    if (need_store) {
      y[store_offset + i] = static_cast<Ty>(reduce_var);
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
                                      int blocking_size,
                                      const kps::DimConfig dim) {
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
  const Tx* input = x + block_offset;
  Tx reduce_input;
  for (; idx < size; idx += stride) {
    MPType reduce_var = init;
    MPType reduce_compute = init;
    for (int loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
      kps::ReadData<Tx, Tx, 1, 1, 1, false>(&reduce_input,
                                            input + loop_idx * left_num + idx,
                                            block.BlockDimX(),
                                            1,
                                            1,
                                            left_num);
      kps::ElementwiseUnary<Tx, MPType, REDUCE_VEC_SIZE, 1, 1, TransformOp>(
          &reduce_compute, &reduce_input, transformer);
      kps::Reduce<MPType,
                  1,
                  1,
                  1,
                  ReduceOp,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &reduce_compute, reducer, false);
    }
    Ty result = static_cast<Ty>(reduce_var);
    kps::WriteData<Ty, 1, 1, 1, false>(
        y + store_offset + idx, &result, block.BlockDimX());
  }

  if (idx < left_num) {
    MPType reduce_var = init;
    MPType reduce_compute = init;
    for (int loop_idx = 0; loop_idx < loop_size; ++loop_idx) {
      kps::ReadData<Tx, Tx, 1, 1, 1, true>(&reduce_input,
                                           input + loop_idx * left_num + idx,
                                           dim.rem_x,
                                           1,
                                           1,
                                           left_num);
      kps::ElementwiseUnary<Tx, MPType, REDUCE_VEC_SIZE, 1, 1, TransformOp>(
          &reduce_compute, &reduce_input, transformer);
      kps::Reduce<MPType,
                  1,
                  1,
                  1,
                  ReduceOp,
                  kps::details::ReduceMode::kLocalMode>(
          &reduce_var, &reduce_compute, reducer, false);
    }
    Ty result = static_cast<Ty>(reduce_var);
    kps::WriteData<Ty, 1, 1, 1, true>(
        y + store_offset + idx, &result, dim.rem_x);
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
                               gpuStream_t stream,
                               ReduceConfig<Ty> config) {
  if (config.reduce_type == kReduceLastDim) {
    int stride_reduce = 1;
    int stride_left = config.reduce_num;
    // for higher performance
    auto reduce_index_calculator = OneDimIndexCal(stride_reduce);
    auto left_index_calculator = OneDimIndexCal(stride_left);

    kps::DimConfig dim = kps::DimConfig(config.grid.x,
                                        config.grid.y,
                                        config.grid.z,
                                        config.block.x,
                                        config.block.y,
                                        0);
    dim.SetRem(config.reduce_num % config.block.x, 0, 0);

#ifdef PADDLE_WITH_XPU2
    ReduceAnyKernel<Tx,
                    Ty,
                    MPType,
                    ReduceOp,
                    TransformOp,
                    OneDimIndexCal><<<8, 128, stream>>>(x_data,
                                                        config.output_data,
                                                        reducer,
                                                        transform,
                                                        init,
                                                        config.reduce_num,
                                                        config.left_num,
                                                        config.reduce_last_dim,
                                                        reduce_index_calculator,
                                                        left_index_calculator,
                                                        dim);
#else
    ReduceAnyKernel<Tx,
                    Ty,
                    MPType,
                    ReduceOp,
                    TransformOp,
                    OneDimIndexCal><<<config.grid, config.block, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        init,
        config.reduce_num,
        config.left_num,
        config.reduce_last_dim,
        reduce_index_calculator,
        left_index_calculator,
        dim);
#endif

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

#ifdef PADDLE_WITH_XPU2
    ReduceAnyKernel<Tx,
                    Ty,
                    MPType,
                    ReduceOp,
                    TransformOp,
                    IndexCalculator><<<8, 128, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        init,
        config.reduce_num,
        config.left_num,
        config.reduce_last_dim,
        reduce_index_calculator,
        left_index_calculator,
        dim);
#else
    ReduceAnyKernel<Tx,
                    Ty,
                    MPType,
                    ReduceOp,
                    TransformOp,
                    IndexCalculator><<<config.grid, config.block, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        init,
        config.reduce_num,
        config.left_num,
        config.reduce_last_dim,
        reduce_index_calculator,
        left_index_calculator,
        dim);
#endif
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
#ifdef PADDLE_WITH_XPU2
    ReduceHigherDimKernel<Ty,
                          Ty,
                          MPType,
                          ReduceOp,
                          kps::IdentityFunctor<Ty, MPType>><<<8, 128, stream>>>(
        config.output_data,
        y_data,
        reducer,
        kps::IdentityFunctor<Ty, MPType>(),
        init,
        config.grid.y,
        config.left_num,
        config.grid.y,
        dim);
#else
    ReduceHigherDimKernel<
        Ty,
        Ty,
        MPType,
        ReduceOp,
        kps::IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
        config.output_data,
        y_data,
        reducer,
        kps::IdentityFunctor<Ty, MPType>(),
        init,
        config.grid.y,
        config.left_num,
        config.grid.y,
        dim);
#endif
  }
}

template <typename Tx,
          typename Ty,
          template <typename> class ReduceOp,
          typename TransformOp>
static
    typename std::enable_if<!std::is_same<Tx, paddle::platform::float16>::value,
                            void>::type
    CubTensorReduceFunctorImpl(const Tx* x_data,
                               Ty* y_data,
                               const TransformOp& transform,
                               int reduce_num,
                               const paddle::platform::Place& place,
                               gpuStream_t stream) {
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

  pten::DenseTensor tmp = pten::DenseTensor(
      pten::make_intrusive<paddle::experimental::SharedStorage>(place),
      pten::DenseTensorMeta(pten::DataType::UINT8,
                            paddle::framework::make_ddim(
                                {static_cast<int64_t>(temp_storage_bytes)})));

  auto* temp_storage = tmp.mutable_data<uint8_t>();

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
static
    typename std::enable_if<std::is_same<Tx, paddle::platform::float16>::value,
                            void>::type
    CubTensorReduceFunctorImpl(const Tx* x_data,
                               Ty* y_data,
                               const TransformOp& transform,
                               int reduce_num,
                               const paddle::platform::Place& place,
                               gpuStream_t stream) {
  PADDLE_THROW(paddle::platform::errors::InvalidArgument(
      "Tx should not be float16 when using cub::DeviceReduce::Reduce()."));
}

template <typename Tx,
          typename Ty,
          template <typename> class ReduceOp,
          typename TransformOp>
void TensorReduceFunctorImpl(const pten::DenseTensor& x,
                             pten::DenseTensor* y,
                             const TransformOp& transform,
                             const std::vector<int>& origin_reduce_dims,
                             gpuStream_t stream) {
  // Allocate memory
  y->mutable_data<Ty>();

  auto x_dim = paddle::framework::vectorize<int>(x.dims());
  auto config = ReduceConfig<Ty>(origin_reduce_dims, x_dim);
  config.Run();
  int numel = x.numel();
  // after config.run()
  // SetOutputData for ReduceHigherDim when should_reduce_again is true,
  // temp_output should be stored temp_data in output_data space or stored in
  // y_data;

  pten::DDim tmp_ddim;
  pten::DenseTensor tmp = pten::DenseTensor(
      pten::make_intrusive<paddle::experimental::SharedStorage>(y->place()),
      pten::DenseTensorMeta(y->dtype(), tmp_ddim, y->layout()));

  auto x_data = x.data<Tx>();
  auto y_data = y->mutable_data<Ty>();

  auto* dev_ctx = static_cast<paddle::platform::CUDADeviceContext*>(
      paddle::platform::DeviceContextPool::Instance().Get(x.place()));
  if (config.reduce_num == 1) {
    std::vector<const DenseTensor*> inputs = {&x};
    std::vector<DenseTensor*> outputs = {y};
    funcs::LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kUnary, Tx, Ty>(
        *dev_ctx, inputs, &outputs, transform);
    return;
  }

  config.SetOutputData(y_data, x.place(), &tmp);
  constexpr bool kIsTxFP16 = std::is_same<Tx, paddle::platform::float16>::value;
  bool use_cub_reduce = config.reduce_num == numel && !kIsTxFP16;
  if (use_cub_reduce) {
    CubTensorReduceFunctorImpl<Tx, Ty, ReduceOp, TransformOp>(
        x_data, y_data, transform, config.reduce_num, x.place(), stream);
    return;
  }

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

#ifdef PADDLE_WITH_XPU2
    ReduceHigherDimKernel<Tx,
                          Ty,
                          MPType,
                          ReduceOp<MPType>,
                          TransformOp><<<8, 128, stream>>>(x_data,
                                                           config.output_data,
                                                           reducer,
                                                           transform,
                                                           reducer.initial(),
                                                           config.reduce_num,
                                                           config.left_num,
                                                           config.blocking_size,
                                                           dim);
#else
    ReduceHigherDimKernel<
        Tx,
        Ty,
        MPType,
        ReduceOp<MPType>,
        TransformOp><<<config.grid, config.block, 0, stream>>>(
        x_data,
        config.output_data,
        reducer,
        transform,
        reducer.initial(),
        config.reduce_num,
        config.left_num,
        config.blocking_size,
        dim);
#endif

    if (config.should_reduce_again) {
      dim3 block = dim3(config.block.x, 1, 1);
      dim3 grid = dim3(config.grid.x, 1, config.grid.z);
      kps::DimConfig dim2 =
          kps::DimConfig(grid.x, grid.y, grid.z, block.x, config.grid.y, 0);
      dim2.SetRem(config.left_num % config.block.x, 0, 0);

#ifdef PADDLE_WITH_XPU2
      ReduceHigherDimKernel<
          Ty,
          Ty,
          MPType,
          ReduceOp<MPType>,
          kps::IdentityFunctor<Ty, MPType>><<<8, 128, stream>>>(
          config.output_data,
          y_data,
          reducer,
          kps::IdentityFunctor<Ty, MPType>(config.grid.y),
          reducer.initial(),
          config.grid.y,
          config.left_num,
          config.grid.y,
          dim2);
#else
      ReduceHigherDimKernel<
          Ty,
          Ty,
          MPType,
          ReduceOp<MPType>,
          kps::IdentityFunctor<Ty, MPType>><<<grid, block, 0, stream>>>(
          config.output_data,
          y_data,
          reducer,
          kps::IdentityFunctor<Ty, MPType>(config.grid.y),
          reducer.initial(),
          config.grid.y,
          config.left_num,
          config.grid.y,
          dim2);
#endif
    }
    return;
  }

  // when reduce_dim.size() == 1 and reduce_dim[0] == x_dim.size() - 1, or
  // when reduce_dim.size() != 1 and reduce_dim.size() != x_dim.size(), this
  // function will be used
  LaunchReduceKernel<Tx, Ty, MPType, ReduceOp<MPType>, TransformOp>(
      x_data, y_data, reducer, transform, reducer.initial(), stream, config);
}

}  // namespace kernels

template <typename T,
          template <typename> class ReduceOp,
          template <typename, typename> class TransformOp>
void Reduce(const GPUContext& dev_ctx,
            const DenseTensor& x,
            bool reduce_all,
            const std::vector<int64_t>& dims,
            bool keep_dim,
            DataType out_dtype,
            DenseTensor* out) {
  std::vector<int> reduce_dims =
      pten::kernels::details::GetReduceDim(dims, x.dims().size(), reduce_all);

  int reduce_num = 1;
  for (auto i : reduce_dims) {
    reduce_num *= (x.dims())[i];
  }

  gpuStream_t stream = dev_ctx.stream();

  if (out_dtype != pten::DataType::UNDEFINED && out_dtype != x.dtype()) {
    PD_DISPATCH_FLOATING_AND_COMPLEX_AND_2_TYPES(
        pten::DataType::INT32,
        pten::DataType::INT64,
        out_dtype,
        "TensorReduceFunctorImpl",
        ([&] {
          using MPType = typename kps::details::MPTypeTrait<data_t>::Type;
          pten::kernels::TensorReduceFunctorImpl<T,
                                                 data_t,
                                                 ReduceOp,
                                                 TransformOp<T, MPType>>(
              x, out, TransformOp<T, MPType>(reduce_num), reduce_dims, stream);
        }));
  } else {
    using MPType = typename kps::details::MPTypeTrait<T>::Type;
    pten::kernels::
        TensorReduceFunctorImpl<T, T, ReduceOp, TransformOp<T, MPType>>(
            x, out, TransformOp<T, MPType>(reduce_num), reduce_dims, stream);
  }
}
}  // namespace pten

#endif
