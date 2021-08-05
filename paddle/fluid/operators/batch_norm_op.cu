/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <cfloat>
#include <string>
#include <vector>
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/norm_utils.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/fluid/platform/float16.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using BatchNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, framework::DataLayout layout>
static __global__ void BNForwardInference(
    const T *x, const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias, const int C, const int N, const int HxW,
    const double epsilon, T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num = N * C * HxW;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> x_sub_mean =
        static_cast<BatchNormParamType<T>>(x[i]) - mean[c];
    BatchNormParamType<T> inv_var = 1 / sqrt(variance[c] + epsilon);
    y[i] = static_cast<T>(scale[c] * x_sub_mean * inv_var + bias[c]);
  }
}

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNForwardTraining(
    const T *x, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias, const int C, const int N, const int HxW,
    const double epsilon, double exponentialAverageFactor, T *y,
    BatchNormParamType<T> *mean, BatchNormParamType<T> *variance,
    BatchNormParamType<T> *save_mean,
    BatchNormParamType<T> *save_inv_variance) {
  int outer_size = C;
  int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storeage;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> variance_val;
  __shared__ BatchNormParamType<T> inv_var_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_i = static_cast<BatchNormParamType<T>>(x[index]);
      x_sum += x_i;
      x_square_sum += x_i * x_i;
    }
    x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
    x_square_sum =
        BlockReduce(variance_storeage).Reduce(x_square_sum, cub::Sum());
    if (threadIdx.x == 0) {
      mean_val = x_sum / inner_size;
      variance_val = x_square_sum / inner_size - mean_val * mean_val;
      inv_var_val = 1 / sqrt(variance_val + epsilon);

      if (save_mean && save_inv_variance) {
        save_mean[i] = mean_val;
        save_inv_variance[i] = inv_var_val;
      }
      mean[i] = (1 - exponentialAverageFactor) * mean_val +
                exponentialAverageFactor * mean[i];
      variance[i] = (1 - exponentialAverageFactor) * variance_val +
                    exponentialAverageFactor * variance[i];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[index]) - mean_val;
      y[index] = scale[i] * x_sub_mean * inv_var_val + bias[i];
    }
  }
}

// The computation pattern is similar to column reduce of matrix.
// Task distribution strategy:
//    1) each thread is respondable for computation of one resultant element
//    (ensure coalesced gloabl mem access)
//    2) when lack paralleism degree and workload of each thread is too much,
//    use 2D grid partition.
void SetBNConfigForCLast(const int max_threads, const int reduce_num,
                         const int left_num, int *blocking_size,
                         bool *should_reduce_again, dim3 *block_dim,
                         dim3 *grid_dim) {
  block_dim->z = 1;
  grid_dim->z = 1;
  *should_reduce_again = false;

  int num_block = (max_threads / left_num);
  if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
    *blocking_size = detail::GetLastPow2(reduce_num / num_block);

    if (*blocking_size <= 1) {
      *blocking_size = detail::GetLastPow2(sqrt(reduce_num));
    } else if (*blocking_size * 2 < reduce_num) {
      *blocking_size *= 2;
    }

    *should_reduce_again = true;

    block_dim->x = 32;
    block_dim->y = 1;
    grid_dim->x = (left_num + block_dim->x - 1) / block_dim->x;
    grid_dim->y = (reduce_num + *blocking_size - 1) / *blocking_size;

  } else {
    block_dim->x = 32;
    *blocking_size = reduce_num;
    grid_dim->x = (left_num + block_dim->x - 1) / block_dim->x;
    grid_dim->y = 1;
  }
}

template <typename T>
__global__ void BnFwCLastSinglePass(
    const T *x, int reduce_num, int left_num,
    const BatchNormParamType<T> *scale, const BatchNormParamType<T> *bias,
    const double epsilon, double exponentialAverageFactor,
    BatchNormParamType<T> *mean, BatchNormParamType<T> *variance,
    BatchNormParamType<T> *saved_mean,
    BatchNormParamType<T> *saved_inv_variance, T *y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
  BatchNormParamType<T> x_square_sum = static_cast<BatchNormParamType<T>>(0);
  BatchNormParamType<T> mean_val = static_cast<BatchNormParamType<T>>(0);
  BatchNormParamType<T> var_val = static_cast<BatchNormParamType<T>>(0);
  BatchNormParamType<T> inv_var_val = static_cast<BatchNormParamType<T>>(0);

  if (idx < left_num) {
    for (int iy = 0; iy < reduce_num; iy++) {
      int id = iy * left_num + idx;
      BatchNormParamType<T> x_val = static_cast<BatchNormParamType<T>>(x[id]);
      x_sum += x_val;
      x_square_sum += x_val * x_val;
    }

    mean_val = x_sum / reduce_num;
    var_val = x_square_sum / reduce_num - mean_val * mean_val;
    inv_var_val = 1 / sqrt(var_val + epsilon);

    if (saved_mean && saved_inv_variance) {
      saved_mean[idx] = mean_val;
      saved_inv_variance[idx] = inv_var_val;
    }
    mean[idx] = (1 - exponentialAverageFactor) * mean_val +
                exponentialAverageFactor * mean[idx];
    variance[idx] = (1 - exponentialAverageFactor) * var_val +
                    exponentialAverageFactor * variance[idx];

    for (int iy = 0; iy < reduce_num; iy++) {
      int id = iy * left_num + idx;
      BatchNormParamType<T> x_sub_mean =
          static_cast<BatchNormParamType<T>>(x[id]) - mean_val;
      y[id] = scale[idx] * x_sub_mean * inv_var_val + bias[idx];
    }
  }
}

template <typename T>
__global__ void BnFwCLast2DReduce(const T *x, int reduce_num, int left_num,
                                  int workload_per_thread, const double epsilon,
                                  double exponentialAverageFactor,
                                  BatchNormParamType<T> *temp_x_sum,
                                  BatchNormParamType<T> *temp_square_sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * workload_per_thread;

  T x_val;
  BatchNormParamType<T> x_sum = 0;
  BatchNormParamType<T> x_square_sum = 0;

  if (idx < left_num) {
    // The workload of the last bid.y may be less than workload_per_thread.
    int loop = reduce_num - idy;
    loop = loop > workload_per_thread ? workload_per_thread : loop;

    for (int iy = 0; iy < loop; iy++) {
      int id = (idy + iy) * left_num + idx;
      BatchNormParamType<T> x_val = static_cast<BatchNormParamType<T>>(x[id]);
      x_sum += x_val;
      x_square_sum += x_val * x_val;
    }
    temp_x_sum[idx + blockIdx.y * left_num] = x_sum;
    temp_square_sum[idx + blockIdx.y * left_num] = x_square_sum;
  }
}

template <typename T>
__global__ void BnFwCLast1DReduce(const BatchNormParamType<T> *temp_mean,
                                  const BatchNormParamType<T> *temp_var,
                                  int workload_per_thread, int left_num,
                                  int global_reduce_num, const double epsilon,
                                  double exponentialAverageFactor,
                                  BatchNormParamType<T> *mean,
                                  BatchNormParamType<T> *variance,
                                  BatchNormParamType<T> *saved_mean,
                                  BatchNormParamType<T> *saved_inv_variance) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  BatchNormParamType<T> x_sum = 0;
  BatchNormParamType<T> x_square_sum = 0;
  BatchNormParamType<T> mean_val = 0;
  BatchNormParamType<T> var_val = 0;
  BatchNormParamType<T> inv_var_val = 0;

  if (idx < left_num) {
    for (int iy = 0; iy < workload_per_thread; iy++) {
      int id = iy * left_num + idx;
      x_sum += static_cast<BatchNormParamType<T>>(temp_mean[id]);
      x_square_sum += static_cast<BatchNormParamType<T>>(temp_var[id]);
    }

    mean_val = x_sum / global_reduce_num;
    var_val = x_square_sum / global_reduce_num - mean_val * mean_val;
    inv_var_val = 1 / sqrt(var_val + epsilon);

    if (saved_mean && saved_inv_variance) {
      saved_mean[idx] = mean_val;
      saved_inv_variance[idx] = inv_var_val;
    }
    mean[idx] = (1 - exponentialAverageFactor) * mean_val +
                exponentialAverageFactor * mean[idx];
    variance[idx] = (1 - exponentialAverageFactor) * var_val +
                    exponentialAverageFactor * variance[idx];
  }
}

template <typename T, framework::DataLayout layout>
static __global__ void BNFwElementwiselyUpdateY(
    const T *x, const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *bias, const int C, const int N, const int HxW,
    const double epsilon, T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int num = N * C * HxW;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> x_sub_mean =
        static_cast<BatchNormParamType<T>>(x[i]) - mean[c];
    y[i] = static_cast<T>(scale[c] * x_sub_mean * variance[c] + bias[c]);
  }
}

template <typename T>
void FastBNForwardTrainingCLast(
    gpuStream_t stream, framework::DataLayout compute_format, Tensor tensor_x,
    const int x_numel, const int max_threads, const T *x,
    const BatchNormParamType<T> *scale, const BatchNormParamType<T> *bias,
    const int C, const int N, const int HxW, const double epsilon,
    double this_factor, T *y, BatchNormParamType<T> *mean_out,
    BatchNormParamType<T> *variance_out, BatchNormParamType<T> *saved_mean,
    BatchNormParamType<T> *saved_inv_variance) {
  // set configs.
  dim3 block;
  dim3 grid;
  bool should_reduce_again = false;
  int blocking_size = 1;
  int reduce_num = N * HxW;
  int left_num = C;
  SetBNConfigForCLast(max_threads, reduce_num, left_num, &blocking_size,
                      &should_reduce_again, &block, &grid);

  if (!should_reduce_again) {
    BnFwCLastSinglePass<T><<<grid, block, 0, stream>>>(
        x, reduce_num, left_num, scale, bias, epsilon, this_factor, mean_out,
        variance_out, saved_mean, saved_inv_variance, y);
  } else {
    // malloc for temp memory.
    framework::Tensor tmp_mean;
    framework::Tensor tmp_var;
    tmp_mean.mutable_data<BatchNormParamType<T>>(
        framework::make_ddim({static_cast<int64_t>(
            left_num * grid.y * sizeof(BatchNormParamType<T>))}),
        tensor_x.place());

    tmp_var.mutable_data<BatchNormParamType<T>>(
        framework::make_ddim({static_cast<int64_t>(
            left_num * grid.y * sizeof(BatchNormParamType<T>))}),
        tensor_x.place());

    // step-1: column reduce with 2D task distribution.
    BnFwCLast2DReduce<T><<<grid, block, 0, stream>>>(
        x, reduce_num, left_num, blocking_size, epsilon, this_factor,
        tmp_mean.template data<BatchNormParamType<T>>(),
        tmp_var.template data<BatchNormParamType<T>>());

    // step-2: column reduce with 1D task distribution.
    BnFwCLast1DReduce<T><<<grid.x, block.x, 0, stream>>>(
        tmp_mean.template data<BatchNormParamType<T>>(),
        tmp_var.template data<BatchNormParamType<T>>(), grid.y, left_num,
        reduce_num, epsilon, this_factor, mean_out, variance_out, saved_mean,
        saved_inv_variance);

    // step-3: elementwisely update y:
    // y = scale * (x - mean) / sqrt(var) + bias
    const int block_size = 256;
    const int grid_size = (N * C * HxW + block_size - 1) / block_size;
    BNFwElementwiselyUpdateY<
        T, DataLayout::kNHWC><<<grid_size, block_size, 0, stream>>>(
        x, saved_mean, saved_inv_variance, scale, bias, C, N, HxW, epsilon, y);
  }
}

void SetBNConfigForNCHW(const int max_threads, const int reduce_num,
                        const int left_num, bool *should_reduce_again,
                        dim3 *block_dim, dim3 *grid_dim) {
  constexpr int min_reduce_num_per_thread = 16;
  constexpr int max_reduce_num_per_thread = 256;
  constexpr int max_num_threads = detail::kMaxThread;

  int grid_num, reduce_num_per_thread;
  block_dim->x = detail::GetBlockDim(reduce_num);
  block_dim->y = 1;
  grid_num = left_num;
  reduce_num_per_thread =
      detail::AlignUp(reduce_num, block_dim->x * block_dim->y);

  int num_threads = block_dim->x * block_dim->y;
  int max_num_blocks = max_threads / num_threads;

  // set grid size.
  // Whether to set grid.y larger than 1, there are 3 following rules:
  // 1. The number that each thread process should no less than
  //    min_reduce_num_per_thread but no more than max_reduce_num_per_thread;
  // 2. It should maximize the utilization of SM.
  // So we choose the minimum between input_split_num_1 and input_split_num_3
  // to make each thread process as mush data as possible. Meanwhile,
  // the number cannot be larger than max_reduce_num_per_thread, so we
  // choose the maximum between the result above and input_split_num_2.
  int input_split_num_1 =
      detail::AlignUp(reduce_num_per_thread, min_reduce_num_per_thread);
  int input_split_num_2 =
      detail::AlignUp(reduce_num_per_thread, max_reduce_num_per_thread);
  int input_split_num_3 = detail::AlignUp(max_num_blocks, grid_num);

  grid_dim->x = grid_num;
  grid_dim->y = std::max(std::min(input_split_num_1, input_split_num_3),
                         input_split_num_2);
  // if grid.y > 1, we need launch another kernel again due to block sync.
  if (grid_dim->y > 1) {
    *should_reduce_again = true;
  }
}

struct BnNCHWIndexCalculator {
  BnNCHWIndexCalculator(int N, int C, int HxW) {
    CxHxW = C * HxW;
    divmoder = FastDivMod(HxW);
  }

  __device__ inline int Get(int offset) const {
    int index = 0;
    auto divmod = divmoder.Divmod(offset);
    index = divmod.val[0] * CxHxW + divmod.val[1];
    return index;
  }
  int CxHxW;
  FastDivMod divmoder;
};

template <typename Tx>
__global__ void BnFwNCHWSinglePass(
    const Tx *x, int reduce_num, int left_num, int C, int HxW,
    const BatchNormParamType<Tx> *scale, const BatchNormParamType<Tx> *bias,
    const double epsilon, double exponentialAverageFactor,
    BatchNormParamType<Tx> *mean, BatchNormParamType<Tx> *variance,
    BatchNormParamType<Tx> *saved_mean,
    BatchNormParamType<Tx> *saved_inv_variance, Tx *y,
    BnNCHWIndexCalculator bn_index_calculator) {
  int input_idx, left_idx, stride;
  input_idx = threadIdx.x;
  left_idx = blockIdx.x;
  stride = blockDim.x;

  // calculate the offset, the addr where each thread really start.
  int input_offset = left_idx * HxW;
  const Tx *input = x + input_offset;

  int CxHxW = C * HxW;

  Tx x_val;
  BatchNormParamType<Tx> mean_sum = 0;
  BatchNormParamType<Tx> square_sum = 0;
  BatchNormParamType<Tx> mean_val = 0;
  BatchNormParamType<Tx> var_val = 0;
  BatchNormParamType<Tx> inv_var_val = 0;
  __shared__ BatchNormParamType<Tx> share_inv_var_val;
  __shared__ BatchNormParamType<Tx> share_mean_val;

  auto reducer = CustomSum<BatchNormParamType<Tx>>();

  // 1. reduce for each thread
  if (left_idx < left_num) {
    // 1.1. load REDUCE_VEC_SIZE data once, and then compute
    BatchNormParamType<Tx> input_reg[REDUCE_VEC_SIZE];
    int bound = reduce_num - (REDUCE_VEC_SIZE - 1) * stride;
    while (input_idx < bound) {
#pragma unroll
      for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
        int reduce_idx = input_idx + i * stride;
        // Note: when using fastdivmod optimization, uncomment 1st line, comment
        // 2-5 lines.
        // int idx_x = bn_index_calculator.Get(reduce_idx);
        int div = reduce_idx / (HxW);  // get batch id
        int mod = reduce_idx % (HxW);  // get offset in HW orientation
        int idx_x = div * CxHxW + mod;
        input_reg[i] = static_cast<BatchNormParamType<Tx>>(input[idx_x]);
      }
#pragma unroll
      for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
        mean_sum += input_reg[i];
        square_sum += input_reg[i] * input_reg[i];
      }
      input_idx += REDUCE_VEC_SIZE * stride;
    }

    // 1.2. deal with the remain part
    int input_idx_tmp = input_idx;
#pragma unroll
    for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
      if (input_idx >= reduce_num) {
        break;
      }
      int reduce_idx = input_idx;
      // Note: when using fastdivmod optimization, uncomment 1st line, comment
      // 2-5 lines.
      // int idx_x = bn_index_calculator.Get(reduce_idx);
      int div = reduce_idx / (HxW);
      int mod = reduce_idx % (HxW);
      int idx_x = div * CxHxW + mod;
      input_reg[i] = static_cast<BatchNormParamType<Tx>>(input[idx_x]);
      input_idx += stride;
    }
    input_idx = input_idx_tmp;
#pragma unroll
    for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
      if (input_idx >= reduce_num) {
        break;
      }
      mean_sum += input_reg[i];
      square_sum += input_reg[i] * input_reg[i];
      input_idx += stride;
    }
  }

  // 2. reduce in block x
  mean_sum = BlockXReduce(mean_sum, reducer);
  square_sum = BlockXReduce(square_sum, reducer);

  if (left_idx < left_num) {
    if (threadIdx.x == 0) {
      mean_val = mean_sum / reduce_num;
      var_val = square_sum / reduce_num - mean_val * mean_val;
      inv_var_val = 1 / sqrt(var_val + epsilon);
      share_mean_val = mean_val;
      share_inv_var_val = inv_var_val;

      if (saved_mean && saved_inv_variance) {
        saved_mean[left_idx] = mean_val;
        saved_inv_variance[left_idx] = inv_var_val;
      }
      mean[left_idx] = (1 - exponentialAverageFactor) * mean_val +
                       exponentialAverageFactor * mean[left_idx];
      variance[left_idx] = (1 - exponentialAverageFactor) * var_val +
                           exponentialAverageFactor * variance[left_idx];
    }
    __syncthreads();
  }

  // 3. loops every elements and update y.
  input_idx = threadIdx.x;
  Tx *output = y + input_offset;

  if (left_idx < left_num) {
    while (input_idx < reduce_num) {
      // Note: when using fastdivmod optimization, uncomment 1st line, comment
      // 2-5 lines.
      // int idx_x = bn_index_calculator.Get(input_idx);
      int div = input_idx / (HxW);
      int mod = input_idx % (HxW);
      int idx_x = div * CxHxW + mod;

      BatchNormParamType<Tx> x_sub_mean =
          static_cast<BatchNormParamType<Tx>>(input[idx_x]) - share_mean_val;
      output[idx_x] =
          scale[left_idx] * x_sub_mean * share_inv_var_val + bias[left_idx];

      input_idx += stride;
    }
  }
}

template <typename Tx>
__global__ void BnFwNCHW2DReduce(const Tx *x, int reduce_num, int left_num,
                                 int C, int HxW, const double epsilon,
                                 double exponentialAverageFactor,
                                 BatchNormParamType<Tx> *temp_x_sum,
                                 BatchNormParamType<Tx> *temp_square_sum,
                                 BnNCHWIndexCalculator bn_index_calculator) {
  int input_idx = blockIdx.y * blockDim.x + threadIdx.x;
  int left_idx = blockIdx.x;
  int stride = gridDim.y * blockDim.x;

  // calculate the offset, the addr where each thread really start.
  int input_offset = left_idx * HxW;
  const Tx *input = x + input_offset;
  int CxHxW = C * HxW;

  Tx x_val;
  BatchNormParamType<Tx> x_sum = 0;
  BatchNormParamType<Tx> x_square_sum = 0;

  auto reducer = CustomSum<BatchNormParamType<Tx>>();

  // 1. reduce for each thread
  if (left_idx < left_num) {
    // 1.1. load REDUCE_VEC_SIZE data once, and then compute
    BatchNormParamType<Tx> input_reg[REDUCE_VEC_SIZE];
    int bound = reduce_num - (REDUCE_VEC_SIZE - 1) * stride;
    while (input_idx < bound) {
#pragma unroll
      for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
        int reduce_idx = input_idx + i * stride;
        // Note: when using fastdivmod optimization, uncomment 1st line, comment
        // 2-5 lines.
        // int idx_x = bn_index_calculator.Get(reduce_idx);
        int div = reduce_idx / (HxW);
        int mod = reduce_idx % (HxW);
        int idx_x = div * CxHxW + mod;
        input_reg[i] = static_cast<BatchNormParamType<Tx>>(input[idx_x]);
      }
#pragma unroll
      for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
        x_sum = x_sum + input_reg[i];
        x_square_sum = x_square_sum + input_reg[i] * input_reg[i];
      }
      input_idx += REDUCE_VEC_SIZE * stride;
    }

    // 1.2. deal with the remain part
    int input_idx_tmp = input_idx;
#pragma unroll
    for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
      if (input_idx >= reduce_num) {
        break;
      }
      int reduce_idx = input_idx;
      // Note: when using fastdivmod optimization, uncomment 1st line, comment
      // 2-5 lines.
      // int idx_x = bn_index_calculator.Get(reduce_idx);
      int div = reduce_idx / (HxW);
      int mod = reduce_idx % (HxW);
      int idx_x = div * CxHxW + mod;
      input_reg[i] = static_cast<BatchNormParamType<Tx>>(input[idx_x]);
      input_idx += stride;
    }
    input_idx = input_idx_tmp;
#pragma unroll
    for (int i = 0; i < REDUCE_VEC_SIZE; ++i) {
      if (input_idx >= reduce_num) {
        break;
      }
      x_sum = x_sum + input_reg[i];
      x_square_sum = x_square_sum + input_reg[i] * input_reg[i];
      input_idx += stride;
    }
  }

  // 2. reduce in block y
  if (blockDim.y > 1) {
    x_sum = BlockYReduce(x_sum, reducer);
    x_square_sum = BlockYReduce(x_square_sum, reducer);
  }
  __syncthreads();

  // 3. reduce in block x
  x_sum = BlockXReduce(x_sum, reducer);
  x_square_sum = BlockXReduce(x_square_sum, reducer);
  if (threadIdx.x == 0) {
    temp_x_sum[blockIdx.x + blockIdx.y * gridDim.x] = x_sum;
    temp_square_sum[blockIdx.x + blockIdx.y * gridDim.x] = x_square_sum;
  }
}

template <typename T>
void FastBNForwardTrainingNCHW(
    gpuStream_t stream, framework::DataLayout compute_format, Tensor tensor_x,
    const int x_numel, const int max_threads, const T *x,
    const BatchNormParamType<T> *scale, const BatchNormParamType<T> *bias,
    const int C, const int N, const int HxW, const double epsilon,
    double this_factor, T *y, BatchNormParamType<T> *mean_out,
    BatchNormParamType<T> *variance_out, BatchNormParamType<T> *saved_mean,
    BatchNormParamType<T> *saved_inv_variance) {
  // set configs.
  dim3 block;
  dim3 grid;
  bool should_reduce_again = false;
  int reduce_num = N * HxW;
  int left_num = C;
  SetBNConfigForNCHW(max_threads, reduce_num, left_num, &should_reduce_again,
                     &block, &grid);

  auto bn_index_calculator = BnNCHWIndexCalculator(N, C, HxW);

  if (!should_reduce_again) {
    BnFwNCHWSinglePass<T><<<grid, block, 0, stream>>>(
        x, reduce_num, left_num, C, HxW, scale, bias, epsilon, this_factor,
        mean_out, variance_out, saved_mean, saved_inv_variance, y,
        bn_index_calculator);
  } else {
    framework::Tensor tmp_mean;
    framework::Tensor tmp_var;

    tmp_mean.mutable_data<BatchNormParamType<T>>(
        framework::make_ddim({static_cast<int64_t>(
            left_num * grid.y * sizeof(BatchNormParamType<T>))}),
        tensor_x.place());
    tmp_var.mutable_data<BatchNormParamType<T>>(
        framework::make_ddim({static_cast<int64_t>(
            left_num * grid.y * sizeof(BatchNormParamType<T>))}),
        tensor_x.place());

    // step-1: row reduce pattern (strided row).
    BnFwNCHW2DReduce<T><<<grid, block, 0, stream>>>(
        x, reduce_num, left_num, C, HxW, epsilon, this_factor,
        tmp_mean.template data<BatchNormParamType<T>>(),
        tmp_var.template data<BatchNormParamType<T>>(), bn_index_calculator);

    // step-2: column reduce.
    int new_block = 32;
    int new_grid = detail::AlignUp(left_num, 32);
    BnFwCLast1DReduce<T><<<new_grid, new_block, 0, stream>>>(
        tmp_mean.template data<BatchNormParamType<T>>(),
        tmp_var.template data<BatchNormParamType<T>>(), grid.y, left_num,
        reduce_num, epsilon, this_factor, mean_out, variance_out, saved_mean,
        saved_inv_variance);

    // step-3: elementwisely update y.
    // y = scale * (x - mean) / sqrt(var) + bias
    const int block_size = 256;
    const int grid_size = (N * C * HxW + block_size - 1) / block_size;
    BNFwElementwiselyUpdateY<
        T, DataLayout::kNCHW><<<grid_size, block_size, 0, stream>>>(
        x, saved_mean, saved_inv_variance, scale, bias, C, N, HxW, epsilon, y);
  }
}

template <typename T>
void BNForwardTrainingDriver(
    gpuStream_t stream, framework::DataLayout compute_format, Tensor tensor_x,
    const int x_numel, const int max_threads, const T *x,
    const BatchNormParamType<T> *scale, const BatchNormParamType<T> *bias,
    const int C, const int N, const int HxW, const double epsilon,
    double this_factor, T *y, BatchNormParamType<T> *mean_out,
    BatchNormParamType<T> *variance_out, BatchNormParamType<T> *saved_mean,
    BatchNormParamType<T> *saved_variance) {
  int reduce_num = N * HxW;
  int left_num = C;
  bool is_clast = (compute_format == DataLayout::kNCHW && HxW == 1) ||
                  (compute_format == DataLayout::kNHWC);
  bool is_large_enough = (reduce_num > REDUCE_SPLIT_BOUNDARY / 2) ||
                         (left_num > REDUCE_SPLIT_BOUNDARY);

  if (!is_large_enough) {
    int num = x_numel;
    const int block = 256;
    const int max_blocks = std::max(max_threads / block, 1);
    const int grid = std::min(C, max_blocks);
    if (compute_format == DataLayout::kNCHW) {
      BNForwardTraining<T, block,
                        DataLayout::kNCHW><<<grid, block, 0, stream>>>(
          x, scale, bias, C, N, HxW, epsilon, this_factor, y, mean_out,
          variance_out, saved_mean, saved_variance);
    } else {
      BNForwardTraining<T, block,
                        DataLayout::kNHWC><<<grid, block, 0, stream>>>(
          x, scale, bias, C, N, HxW, epsilon, this_factor, y, mean_out,
          variance_out, saved_mean, saved_variance);
    }
  } else if (is_clast) {
    // fast impl for nhwc format.
    FastBNForwardTrainingCLast<T>(stream, compute_format, tensor_x, x_numel,
                                  max_threads, x, scale, bias, C, N, HxW,
                                  epsilon, this_factor, y, mean_out,
                                  variance_out, saved_mean, saved_variance);
  } else {
    // fast impl for nchw format.
    FastBNForwardTrainingNCHW<T>(stream, compute_format, tensor_x, x_numel,
                                 max_threads, x, scale, bias, C, N, HxW,
                                 epsilon, this_factor, y, mean_out,
                                 variance_out, saved_mean, saved_variance);
  }
}

template <typename T>
class BatchNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    bool test_mode = is_test && (!trainable_stats);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5"
            "But received: the size of input's dimensions is [%d]",
            x_dims.size()));

    auto *y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    auto dtype = platform::CudnnDataType<T>::type;

#ifdef PADDLE_WITH_HIP
    auto compute_format = data_layout == DataLayout::kNHWC ? DataLayout::kNHWC
                                                           : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
    const bool fast_nhwc_batch_norm =
        test_mode ||
        (dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent);

    auto compute_format =
        fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
            ? DataLayout::kNHWC
            : DataLayout::kNCHW;
#endif

    Tensor transformed_x(x->type());
    Tensor transformed_y(y->type());
    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform input tensor from NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                           &transformed_x);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                          &transformed_x);
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, y,
                                                           &transformed_y);
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_y.ShareDataWith(*y);
    }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// miopenTensorDescriptor_t data_desc_;
// miopenTensorDescriptor_t bn_param_desc_;
// miopenBatchNormMode_t mode_;

// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
    cudnnTensorDescriptor_t data_desc_;
    cudnnTensorDescriptor_t bn_param_desc_;
    cudnnBatchNormMode_t mode_;

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
#endif

    if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
      LOG(ERROR) << "Provided epsilon is smaller than "
                 << "CUDNN_BN_MIN_EPSILON. Setting it to "
                 << "CUDNN_BN_MIN_EPSILON instead.";
    }
    epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// mode_ = miopenBNSpatial;
#elif CUDNN_VERSION_MIN(7, 0, 1)
    if (FLAGS_cudnn_batchnorm_spatial_persistent) {
      mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    } else if (H == 1 && W == 1) {
      mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    } else {
      mode_ = CUDNN_BATCHNORM_SPATIAL;
    }
#else
    if (H == 1 && W == 1) {
      mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
    } else {
      mode_ = CUDNN_BATCHNORM_SPATIAL;
    }
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

    VLOG(3) << "Setting descriptors.";
    std::vector<int> dims;
    std::vector<int> strides;
    if (compute_format == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * D * C, 1, W * D * C, D * C, C};
    }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// Note: PERSISTENT not implemented for inference
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(
//         bn_param_desc_, data_desc_, test_mode ? miopenBNSpatial : mode_));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
        data_desc_, CudnnDataType<T>::type,
        x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
    // Note: PERSISTENT not implemented for inference
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDeriveBNTensorDescriptor(
            bn_param_desc_, data_desc_,
            test_mode ? CUDNN_BATCHNORM_SPATIAL : mode_));
#endif

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    auto handle = dev_ctx.cudnn_handle();

    // Now, depending on whether we are running test or not, we have two paths.
    // It is training mode when it's not reference AND not using pre-trained
    // model.
    bool training = !test_mode && !use_global_stats;
    if (!training) {
      // only when test we use input to do computation.
      const auto *est_mean = ctx.Input<Tensor>("Mean");
      const auto *est_var = ctx.Input<Tensor>("Variance");
      // Run inference mode.
      PADDLE_ENFORCE_EQ(
          est_mean->dims().size(), 1UL,
          platform::errors::InvalidArgument(
              "The size of mean's dimensions must equal to 1."
              "But received: the size of mean's dimensions mean is [%d],"
              "the dimensions of mean is [%s].",
              est_mean->dims().size(), est_mean->dims()));
      PADDLE_ENFORCE_EQ(
          est_var->dims().size(), 1UL,
          platform::errors::InvalidArgument(
              "The size of variance's dimensions must equal to 1."
              "But received: the size of variance's dimensions is [%d],"
              "the dimensions of variance is [%s].",
              est_var->dims().size(), est_var->dims()));
      PADDLE_ENFORCE_EQ(
          est_mean->dims()[0], C,
          platform::errors::InvalidArgument(
              "The first dimension of mean must equal to the number of "
              "Channels, which is [%d]. But received: the first dimension"
              "of mean is [%d], the dimensions of mean is [%s].",
              C, est_mean->dims()[0], est_mean->dims()));
      PADDLE_ENFORCE_EQ(
          est_var->dims()[0], C,
          platform::errors::InvalidArgument(
              "The first dimension of variance must equal to the number"
              "of Channels, which is [%d]. But received: the first dimension of"
              "variance is [%d], the dimensions of variance is [%s].",
              C, est_var->dims()[0], est_var->dims()));

#ifdef PADDLE_WITH_HIP
      const int block_size = 256;
      const int grid_size = (N * C * H * W * D + block_size - 1) / block_size;
      if (compute_format == DataLayout::kNCHW) {
        BNForwardInference<
            T,
            DataLayout::kNCHW><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            transformed_x.template data<T>(),
            est_mean->template data<BatchNormParamType<T>>(),
            est_var->template data<BatchNormParamType<T>>(),
            scale->template data<BatchNormParamType<T>>(),
            bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
            epsilon, transformed_y.template data<T>());
      } else {
        BNForwardInference<
            T,
            DataLayout::kNHWC><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
            transformed_x.template data<T>(),
            est_mean->template data<BatchNormParamType<T>>(),
            est_var->template data<BatchNormParamType<T>>(),
            scale->template data<BatchNormParamType<T>>(),
            bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
            epsilon, transformed_y.template data<T>());
      }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenBatchNormalizationForwardInference(
//         handle, miopenBNSpatial,
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kOne())),
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kZero())),
//         data_desc_,
//         static_cast<const void *>(transformed_x.template data<T>()),
//         data_desc_,
//         static_cast<void *>(
//             transformed_y.template mutable_data<T>(ctx.GetPlace())),
//         bn_param_desc_,
//         const_cast<void *>(static_cast<const void *>(
//             scale->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             bias->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             est_mean->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             est_var->template data<BatchNormParamType<T>>())),
//         epsilon));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnBatchNormalizationForwardInference(
              handle,
              // Note: PERSISTENT not implemented for inference
              CUDNN_BATCHNORM_SPATIAL, CudnnDataType<T>::kOne(),
              CudnnDataType<T>::kZero(), data_desc_,
              transformed_x.template data<T>(), data_desc_,
              transformed_y.template mutable_data<T>(ctx.GetPlace()),
              bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
              bias->template data<BatchNormParamType<T>>(),
              est_mean->template data<BatchNormParamType<T>>(),
              est_var->template data<BatchNormParamType<T>>(), epsilon));
#endif
    } else {
      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      // Run training mode.
      // obtain running mean and running inv var, and there is no need
      // to initialize them.

      auto *mean_out = ctx.Output<Tensor>("MeanOut");
      auto *variance_out = ctx.Output<Tensor>("VarianceOut");
      mean_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      variance_out->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

      auto *saved_mean = ctx.Output<Tensor>("SavedMean");
      auto *saved_variance = ctx.Output<Tensor>("SavedVariance");
      saved_mean->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      saved_variance->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());

      if ((N * H * W * D) == 1) {
        // Only 1 element in normalization dimension,
        // skip the batch norm calculation, let y = x.
        framework::TensorCopy(*x, ctx.GetPlace(), y);
      } else {
        double this_factor = 1. - momentum;

        bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
        called = true;
        size_t workspace_size = 0;
        size_t reserve_space_size = 0;
        void *reserve_space_ptr = nullptr;
        void *workspace_ptr = nullptr;
        Tensor workspace_tensor;
        // Create reserve space and workspace for batch norm.
        // Create tensor for each batchnorm op, it will be used in the
        // backward. Thus this tensor shouldn't be temp.
        auto *reserve_space = ctx.Output<Tensor>("ReserveSpace");
        PADDLE_ENFORCE_NOT_NULL(
            reserve_space,
            platform::errors::NotFound(
                "The argument ReserveSpace of batch_norm op is not found."));

        // --------------- cudnn batchnorm workspace ---------------
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::
                cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
                    /*handle=*/handle,
                    /*mode=*/mode_,
                    /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*xDesc=*/data_desc_,
                    /*zDesc=*/nullptr,
                    /*yDesc=*/data_desc_,
                    /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                    /*activationDesc=*/nullptr,
                    /*sizeInBytes=*/&workspace_size));

        // -------------- cudnn batchnorm reserve space --------------
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::
                cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
                    /*handle=*/handle,
                    /*mode=*/mode_,
                    /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*activationDesc=*/nullptr,
                    /*xDesc=*/data_desc_,
                    /*sizeInBytes=*/&reserve_space_size));

        reserve_space_ptr = reserve_space->mutable_data(
            ctx.GetPlace(), transformed_x.type(), reserve_space_size);
        workspace_ptr = workspace_tensor.mutable_data(
            ctx.GetPlace(), transformed_x.type(), workspace_size);
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnBatchNormalizationForwardTrainingEx(
                handle, mode_, CUDNN_BATCHNORM_OPS_BN, CudnnDataType<T>::kOne(),
                CudnnDataType<T>::kZero(), data_desc_,
                transformed_x.template data<T>(), nullptr, nullptr, data_desc_,
                transformed_y.template data<T>(), bn_param_desc_,
                scale->template data<BatchNormParamType<T>>(),
                bias->template data<BatchNormParamType<T>>(), this_factor,
                mean_out->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                variance_out->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                epsilon,
                saved_mean->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                saved_variance->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                nullptr, workspace_ptr, workspace_size, reserve_space_ptr,
                reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
        if (!called) {
#ifdef PADDLE_WITH_HIP
          const int num = transformed_x.numel();
          const int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
          BNForwardTrainingDriver<T>(
              dev_ctx.stream(), compute_format, transformed_x, num, max_threads,
              transformed_x.template data<T>(),
              scale->template data<BatchNormParamType<T>>(),
              bias->template data<BatchNormParamType<T>>(), C, N, H * W * D,
              epsilon, this_factor, transformed_y.template data<T>(),
              mean_out->template data<BatchNormParamType<T>>(),
              variance_out->template data<BatchNormParamType<T>>(),
              saved_mean->template data<BatchNormParamType<T>>(),
              saved_variance->template data<BatchNormParamType<T>>());

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenBatchNormalizationForwardTraining(
//         handle, mode_, const_cast<void *>(static_cast<const void *>(
//                            CudnnDataType<T>::kOne())),
//         const_cast<void *>(
//             static_cast<const void *>(CudnnDataType<T>::kZero())),
//         data_desc_,
//         static_cast<const void *>(transformed_x.template data<T>()),
//         data_desc_,
//         static_cast<void *>(
//             transformed_y.template mutable_data<T>(ctx.GetPlace())),
//         bn_param_desc_,
//         const_cast<void *>(static_cast<const void *>(
//             scale->template data<BatchNormParamType<T>>())),
//         const_cast<void *>(static_cast<const void *>(
//             bias->template data<BatchNormParamType<T>>())),
//         this_factor,
//         static_cast<void *>(
//             mean_out->template mutable_data<BatchNormParamType<T>>(
//                 ctx.GetPlace())),
//         static_cast<void *>(variance_out->template mutable_data<
//                             BatchNormParamType<T>>(ctx.GetPlace())),
//         epsilon,
//         static_cast<void *>(
//             saved_mean->template mutable_data<BatchNormParamType<T>>(
//                 ctx.GetPlace())),
//         static_cast<void *>(saved_variance->template mutable_data<
//                             BatchNormParamType<T>>(ctx.GetPlace()))));
#else
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnBatchNormalizationForwardTraining(
                  handle, mode_, CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), data_desc_,
                  transformed_x.template data<T>(), data_desc_,
                  transformed_y.template mutable_data<T>(ctx.GetPlace()),
                  bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
                  bias->template data<BatchNormParamType<T>>(), this_factor,
                  mean_out->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  variance_out->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  epsilon,
                  saved_mean->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  saved_variance->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace())));
#endif
        }
      }
    }

    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW && x_dims.size() > 2) {
      VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
      TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
          ctx, &transformed_y, y);
    }
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// clean when exit.
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
    // clean when exit.
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#endif
  }
};

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void KeBNBackwardScaleBias(
    const T *dy, const T *x, const BatchNormParamType<T> *mean,
    const BatchNormParamType<T> *variance, const double epsilon, const int N,
    const int C, const int HxW, BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    BatchNormParamType<T> inv_var_i = 1.0 / sqrt(variance[i] + epsilon);
    BatchNormParamType<T> mean_i = mean[i];
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += static_cast<BatchNormParamType<T>>(dy[index]) *
                (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
      db_sum += static_cast<BatchNormParamType<T>>(dy[index]);
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum * inv_var_i;
      dbias[i] = db_sum;
    }
    __syncthreads();
  }
}

template <typename T, framework::DataLayout layout>
static __global__ void KeBNBackwardData(const T *dy,
                                        const BatchNormParamType<T> *scale,
                                        const BatchNormParamType<T> *variance,
                                        const double epsilon, const int C,
                                        const int HxW, const int num, T *dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
    BatchNormParamType<T> inv_var = 1.0 / sqrt(variance[c] + epsilon);
    dx[i] = static_cast<T>(static_cast<BatchNormParamType<T>>(dy[i]) *
                           scale[c] * inv_var);
  }
}

template <typename T>
static __global__ void KeBNRestoreData(const framework::DataLayout layout, T *x,
                                       const BatchNormParamType<T> *scale,
                                       const BatchNormParamType<T> *bias,
                                       const BatchNormParamType<T> *mean,
                                       const BatchNormParamType<T> *variance,
                                       double epsilon, int C, int M,
                                       const int num, const T *y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
    const int c = layout == framework::DataLayout::kNCHW ? (i / M) % C : i % C;
    auto y_i = static_cast<BatchNormParamType<T>>(y[i]);
    auto x_i = (y_i - bias[c]) / scale[c] / variance[c] + mean[c];
    x[i] = static_cast<T>(x_i);
  }
}

template <typename T>
class InplaceHelper {
 public:
  void operator()(const framework::DataLayout layout, T *x,
                  const BatchNormParamType<T> *scale,
                  const BatchNormParamType<T> *bias,
                  const BatchNormParamType<T> *mean,
                  const BatchNormParamType<T> *variance, double epsilon, int C,
                  int M, const int num, const T *y, int grid2, const int block,
                  const gpuStream_t &stream) {
    PADDLE_ENFORCE_EQ(x, y, platform::errors::InvalidArgument(
                                "X and Y should be inplaced in inplace mode"));
    KeBNRestoreData<<<grid2, block, 0, stream>>>(
        layout, x, scale, bias, mean, variance, epsilon, C, M, num, y);
  }
};

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackward(
    const T *dy, const T *x, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *saved_mean,
    const BatchNormParamType<T> *saved_inv_variance, const int C, const int N,
    const int HxW, const double epsilon, T *dx, BatchNormParamType<T> *dscale,
    BatchNormParamType<T> *dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;
  __shared__ typename BlockReduce::TempStorage mean_storage;
  __shared__ typename BlockReduce::TempStorage variance_storeage;
  __shared__ BatchNormParamType<T> inv_var_val;
  __shared__ BatchNormParamType<T> mean_val;
  __shared__ BatchNormParamType<T> dscale_val;
  __shared__ BatchNormParamType<T> dbias_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> ds_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> db_sum = static_cast<BatchNormParamType<T>>(0);

    if (saved_mean && saved_inv_variance) {
      if (threadIdx.x == 0) {
        inv_var_val = saved_inv_variance[i];
        mean_val = saved_mean[i];
      }
    } else {
      BatchNormParamType<T> x_sum = static_cast<BatchNormParamType<T>>(0);
      BatchNormParamType<T> x_square_sum =
          static_cast<BatchNormParamType<T>>(0);

      for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        const int index = layout == framework::DataLayout::kNCHW
                              ? (j / HxW * C + i) * HxW + j % HxW
                              : j * outer_size + i;
        BatchNormParamType<T> x_i =
            static_cast<BatchNormParamType<T>>(x[index]);
        x_sum += x_i;
        x_square_sum += x_i * x_i;
      }
      x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
      x_square_sum =
          BlockReduce(variance_storeage).Reduce(x_square_sum, cub::Sum());
      if (threadIdx.x == 0) {
        mean_val = x_sum / inner_size;
        inv_var_val =
            1 / sqrt(x_square_sum / inner_size - mean_val * mean_val + epsilon);
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      ds_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_val);
      db_sum += dy_i;
    }
    ds_sum = BlockReduce(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale_val = ds_sum * inv_var_val;
      dbias_val = db_sum;
      dscale[i] = dscale_val;
      dbias[i] = dbias_val;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] = scale[i] * inv_var_val *
                  (static_cast<BatchNormParamType<T>>(dy[index]) -
                   dbias_val / static_cast<BatchNormParamType<T>>(inner_size) -
                   (static_cast<BatchNormParamType<T>>(x[index]) - mean_val) *
                       inv_var_val * dscale_val / inner_size);
    }
  }
}

template <typename T, int BlockDim, framework::DataLayout layout>
static __global__ LAUNCH_BOUNDS(BlockDim) void BNBackwardData(
    const T *dy, const BatchNormParamType<T> *scale,
    const BatchNormParamType<T> *mean, const T *x,
    const BatchNormParamType<T> *variance, const int C, const int N,
    const int HxW, T *dx) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<BatchNormParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage dy_storage;
  __shared__ typename BlockReduce::TempStorage dy_x_sub_mean_storage;
  __shared__ BatchNormParamType<T> dy_sum_val;
  __shared__ BatchNormParamType<T> dy_x_sub_mean_sum_val;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    BatchNormParamType<T> inv_var_i = variance[i];
    BatchNormParamType<T> mean_i = mean[i];
    BatchNormParamType<T> dy_sum = static_cast<BatchNormParamType<T>>(0);
    BatchNormParamType<T> dy_x_sub_mean_sum =
        static_cast<BatchNormParamType<T>>(0);
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      BatchNormParamType<T> dy_i =
          static_cast<BatchNormParamType<T>>(dy[index]);
      dy_sum += dy_i;
      dy_x_sub_mean_sum +=
          dy_i * (static_cast<BatchNormParamType<T>>(x[index]) - mean_i);
    }

    dy_sum = BlockReduce(dy_storage).Reduce(dy_sum, cub::Sum());
    dy_x_sub_mean_sum = BlockReduce(dy_x_sub_mean_storage)
                            .Reduce(dy_x_sub_mean_sum, cub::Sum());

    if (threadIdx.x == 0) {
      dy_sum_val = dy_sum;
      dy_x_sub_mean_sum_val = dy_x_sub_mean_sum;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = layout == framework::DataLayout::kNCHW
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      dx[index] =
          (static_cast<BatchNormParamType<T>>(dy[index]) -
           dy_sum_val / static_cast<BatchNormParamType<T>>(inner_size) -
           (static_cast<BatchNormParamType<T>>(x[index]) - mean_i) *
               dy_x_sub_mean_sum_val * inv_var_i * inv_var_i / inner_size) *
          scale[i] * inv_var_i;
    }
  }
}

template <typename T>
class BatchNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");

    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    // batch_norm with inplace as false will take X as grad input, which
    // is same as cuDNN batch_norm backward calculation, batch_norm
    // with inplace as true only take Y as input and X should be calculate
    // by inverse operation of batch_norm on Y
    const Tensor *x;
    bool is_inplace;
    if (ctx.HasInput("Y")) {
      x = ctx.Input<Tensor>("Y");
      is_inplace = true;
      if (d_x) {
        PADDLE_ENFORCE_EQ(d_x, d_y,
                          platform::errors::InvalidArgument(
                              "X@GRAD and Y@GRAD not inplace in inplace mode"));
      }
    } else {
      x = ctx.Input<Tensor>("X");
      is_inplace = false;
      if (d_x) {
        PADDLE_ENFORCE_NE(
            d_x, d_y, platform::errors::InvalidArgument(
                          "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
      }
    }

    const bool is_test = ctx.Attr<bool>("is_test");
    use_global_stats = is_test || use_global_stats;

    const auto &x_dims = x->dims();

    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5."
            "But received: the size of input's dimensions is [%d],"
            "the dimensions of input is [%s]",
            x_dims.size(), x_dims));
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    // init output
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
    }

    if (d_scale && d_bias) {
      d_scale->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
      d_bias->mutable_data<BatchNormParamType<T>>(ctx.GetPlace());
    }
    PADDLE_ENFORCE_EQ(
        scale->dims().size(), 1UL,
        platform::errors::InvalidArgument(
            "The size of scale's dimensions must equal to 1. But received: "
            "the size of scale's dimensions is [%d], the dimensions of scale "
            "is [%s].",
            scale->dims().size(), scale->dims()));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::InvalidArgument(
            "The first dimension of scale must equal to Channels[%d]. But "
            "received: the first dimension of scale is [%d]",
            C, scale->dims()[0]));

    auto dtype = platform::CudnnDataType<T>::type;
    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");
#ifdef PADDLE_WITH_HIP
    auto compute_format = data_layout == DataLayout::kNHWC ? DataLayout::kNHWC
                                                           : DataLayout::kNCHW;

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// HIP do not support compute format of NHWC
// auto compute_format = DataLayout::kNCHW;
#else
    const bool fast_nhwc_batch_norm =
        dtype == CUDNN_DATA_HALF && FLAGS_cudnn_batchnorm_spatial_persistent &&
        reserve_space != nullptr;
    auto compute_format =
        fast_nhwc_batch_norm && data_layout == DataLayout::kNHWC
            ? DataLayout::kNHWC
            : DataLayout::kNCHW;
#endif

    Tensor transformed_x(x->type());
    Tensor transformed_d_y(d_y->type());
    Tensor transformed_d_x;
    if (data_layout == DataLayout::kNHWC &&
        compute_format == DataLayout::kNCHW) {
      VLOG(3) << "Transform input tensor from NHWC to NCHW.";
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                           &transformed_x);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, x,
                                                          &transformed_x);
      ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_y,
                                                           &transformed_d_y);
      TransToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_y,
                                                          &transformed_d_y);
      if (d_x) {
        ResizeToChannelFirst<platform::CUDADeviceContext, T>(ctx, d_x,
                                                             &transformed_d_x);
      }
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_d_y.ShareDataWith(*d_y);
      if (d_x) {
        transformed_d_x.ShareDataWith(*d_x);
      }
    }

    std::vector<int> dims;
    std::vector<int> strides;
    if (compute_format == DataLayout::kNCHW) {
      dims = {N, C, H, W, D};
      strides = {C * H * W * D, H * W * D, W * D, D, 1};
    } else {
      dims = {N, C, H, W, D};
      strides = {H * W * C * D, 1, W * D * C, D * C, C};
    }

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const int num = transformed_x.numel();
#ifdef HIPCC
    const int block = 256;
#else
    const int block = 512;
#endif
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid1 = (num + block - 1) / block;
    int grid2 = std::min(C, max_blocks);
    auto stream = dev_ctx.stream();
    InplaceHelper<T> inplace_functor;

    if (!use_global_stats) {
      if ((N * H * W * D) == 1) {
        if (d_x) {
          framework::TensorCopy(*d_y, ctx.GetPlace(), d_x);
        }
        math::SetConstant<platform::CUDADeviceContext, BatchNormParamType<T>>
            functor;
        functor(dev_ctx, d_scale, static_cast<BatchNormParamType<T>>(0));
        functor(dev_ctx, d_bias, static_cast<BatchNormParamType<T>>(0));
        return;
      }

// ------------------- cudnn descriptors ---------------------
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// miopenTensorDescriptor_t data_desc_;
// miopenTensorDescriptor_t bn_param_desc_;
// miopenBatchNormMode_t mode_;

// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenCreateTensorDescriptor(&bn_param_desc_));
#else
      cudnnTensorDescriptor_t data_desc_;
      cudnnTensorDescriptor_t bn_param_desc_;
      cudnnBatchNormMode_t mode_;

      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&data_desc_));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateTensorDescriptor(&bn_param_desc_));
#endif
      if (epsilon <= CUDNN_BN_MIN_EPSILON - FLT_EPSILON) {
        LOG(ERROR) << "Provided epsilon is smaller than "
                   << "CUDNN_BN_MIN_EPSILON. Setting it to "
                   << "CUDNN_BN_MIN_EPSILON instead.";
      }
      epsilon = std::max(epsilon, CUDNN_BN_MIN_EPSILON);
#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// mode_ = miopenBNSpatial;
#elif CUDNN_VERSION_MIN(7, 0, 1)
      if (FLAGS_cudnn_batchnorm_spatial_persistent) {
        mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
      } else if (H == 1 && W == 1) {
        mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
      } else {
        mode_ = CUDNN_BATCHNORM_SPATIAL;
      }
#else
      if (H == 1 && W == 1) {
        mode_ = CUDNN_BATCHNORM_PER_ACTIVATION;
      } else {
        mode_ = CUDNN_BATCHNORM_SPATIAL;
      }
#endif  // CUDNN_VERSION_MIN(7, 0, 1)

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::miopenSetTensorDescriptor(
//     data_desc_, CudnnDataType<T>::type,
//     x_dims.size() > 3 ? x_dims.size() : 4, const_cast<int *>(dims.data()),
//     const_cast<int *>(strides.data())));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDeriveBNTensorDescriptor(bn_param_desc_,
//                                                       data_desc_, mode_));
#else
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetTensorNdDescriptor(
          data_desc_, CudnnDataType<T>::type,
          x_dims.size() > 3 ? x_dims.size() : 4, dims.data(), strides.data()));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDeriveBNTensorDescriptor(bn_param_desc_,
                                                           data_desc_, mode_));
#endif

      const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
      const auto *saved_var = ctx.Input<Tensor>("SavedVariance");
      const auto *saved_mean_data =
          saved_mean->template data<BatchNormParamType<T>>();
      const auto *saved_var_data =
          saved_var->template data<BatchNormParamType<T>>();

      if (is_inplace) {
        inplace_functor(compute_format, transformed_x.data<T>(),
                        scale->template data<BatchNormParamType<T>>(),
                        bias->template data<BatchNormParamType<T>>(),
                        saved_mean_data, saved_var_data, epsilon, C, H * W * D,
                        num, transformed_x.data<T>(), grid2, block, stream);
      }

      // This branch calls CUDNN APIs
      if (d_x && d_scale && d_bias) {
        bool called = false;
#if CUDNN_VERSION_MIN(7, 4, 1)
        called = true;
        size_t workspace_size = 0;
        void *workspace_ptr = nullptr;
        Tensor workspace_tensor;
        auto reserve_space_size = reserve_space->memory_size();
        // --------------- cudnn batchnorm workspace ---------------
        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::
                cudnnGetBatchNormalizationBackwardExWorkspaceSize(
                    /*handle=*/dev_ctx.cudnn_handle(),
                    /*mode=*/mode_,
                    /*bnIps=*/CUDNN_BATCHNORM_OPS_BN,
                    /*xDesc=*/data_desc_,
                    /*yDesc=*/data_desc_,
                    /*dyDesc=*/data_desc_,
                    /*dzDesc=*/nullptr,
                    /*dxDesc=*/data_desc_,
                    /*bnScaleBiasMeanVarDesc=*/bn_param_desc_,
                    /*activationDesc=*/nullptr,
                    /*sizeInBytes=*/&workspace_size));

        workspace_ptr = workspace_tensor.mutable_data(
            ctx.GetPlace(), transformed_x.type(), workspace_size);

        PADDLE_ENFORCE_CUDA_SUCCESS(
            platform::dynload::cudnnBatchNormalizationBackwardEx(
                /*handle=*/dev_ctx.cudnn_handle(),
                /*mode=*/mode_,
                /*bnOps=*/CUDNN_BATCHNORM_OPS_BN,
                /*alphaDataDiff=*/CudnnDataType<T>::kOne(),
                /*betaDataDiff=*/CudnnDataType<T>::kZero(),
                /*alphaParamDiff=*/CudnnDataType<T>::kOne(),
                /*betaParamDiff=*/CudnnDataType<T>::kZero(),
                /*xDesc=*/data_desc_,
                /*xData=*/transformed_x.template data<T>(),
                /*yDesc=*/nullptr,
                /*yData=*/nullptr,
                /*dyDesc=*/data_desc_,
                /*dyData=*/transformed_d_y.template data<T>(),
                /*dzDesc=*/nullptr,
                /*dzData=*/nullptr,
                /*dxDesc=*/data_desc_,
                /*dxData=*/transformed_d_x.template mutable_data<T>(
                    ctx.GetPlace()),
                /*dBnScaleBiasDesc=*/bn_param_desc_,
                /*bnScaleData=*/scale->template data<BatchNormParamType<T>>(),
                /*bnBiasData=*/nullptr,
                /*dBnScaleData=*/d_scale
                    ->template mutable_data<BatchNormParamType<T>>(
                        ctx.GetPlace()),
                /*dBnBiasData=*/d_bias
                    ->template mutable_data<BatchNormParamType<T>>(
                        ctx.GetPlace()),
                /*epsilon=*/epsilon,
                /*savedMean=*/saved_mean_data,
                /*savedInvVariance=*/saved_var_data,
                /*activationDesc=*/nullptr,
                /*workspace=*/workspace_ptr,
                /*workSpaceSizeInBytes=*/workspace_size,
                /*reserveSpace=*/const_cast<T *>(
                    reserve_space->template data<T>()),
                /*reserveSpaceSizeInBytes=*/reserve_space_size));
#endif  // CUDNN_VERSION_MIN(7, 4, 1)
        if (!called) {
#ifdef PADDLE_WITH_HIP
          if (compute_format == DataLayout::kNCHW) {
            BNBackward<
                T, block,
                DataLayout::kNCHW><<<grid2, block, 0, dev_ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(), saved_mean_data,
                saved_var_data, C, N, H * W * D, epsilon,
                transformed_d_x.template data<T>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()));
          } else {
            BNBackward<
                T, block,
                DataLayout::kNHWC><<<grid2, block, 0, dev_ctx.stream()>>>(
                transformed_d_y.template data<T>(),
                transformed_x.template data<T>(),
                scale->template data<BatchNormParamType<T>>(), saved_mean_data,
                saved_var_data, C, N, H * W * D, epsilon,
                transformed_d_x.template data<T>(),
                d_scale->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()),
                d_bias->template mutable_data<BatchNormParamType<T>>(
                    ctx.GetPlace()));
          }

// TODO(wangran16): wait for MIOpen to improve the performance of BN
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenBatchNormalizationBackward(
//         dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
//         CudnnDataType<T>::kZero(), data_desc_,
//         transformed_x.template data<T>(), data_desc_,
//         transformed_d_y.template data<T>(), data_desc_,
//         transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
//         bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
//         d_scale->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         d_bias->template mutable_data<BatchNormParamType<T>>(
//             ctx.GetPlace()),
//         epsilon, saved_mean_data, saved_var_data));
#else
          PADDLE_ENFORCE_CUDA_SUCCESS(
              platform::dynload::cudnnBatchNormalizationBackward(
                  dev_ctx.cudnn_handle(), mode_, CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), CudnnDataType<T>::kOne(),
                  CudnnDataType<T>::kZero(), data_desc_,
                  transformed_x.template data<T>(), data_desc_,
                  transformed_d_y.template data<T>(), data_desc_,
                  transformed_d_x.template mutable_data<T>(ctx.GetPlace()),
                  bn_param_desc_, scale->template data<BatchNormParamType<T>>(),
                  d_scale->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  d_bias->template mutable_data<BatchNormParamType<T>>(
                      ctx.GetPlace()),
                  epsilon, saved_mean_data, saved_var_data));
#endif
        }

        if (data_layout == DataLayout::kNHWC &&
            compute_format == DataLayout::kNCHW) {
          VLOG(3) << "Transform batchnorm output from NCHW to NHWC";
          TransToChannelLast<paddle::platform::CUDADeviceContext, T>(
              ctx, &transformed_d_x, d_x);
        }
      } else {
        // This branch call CUDA kernels
        if (compute_format == DataLayout::kNCHW) {
          if (d_x) {
            BNBackwardData<T, block, framework::DataLayout::kNCHW><<<
                grid2, block, 0, dev_ctx.stream()>>>(
                d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
                saved_mean_data, x->data<T>(), saved_var_data, C, N, H * W * D,
                d_x->data<T>());
          }
          if (d_scale && d_bias) {
            KeBNBackwardScaleBias<
                T, block,
                framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
                d_y->data<T>(), x->data<T>(), saved_mean_data, saved_var_data,
                epsilon, N, C, H * W * D,
                d_scale->data<BatchNormParamType<T>>(),
                d_bias->data<BatchNormParamType<T>>());
          }
        } else {
          if (d_x) {
            BNBackwardData<T, block, framework::DataLayout::kNHWC><<<
                grid2, block, 0, dev_ctx.stream()>>>(
                d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
                saved_mean_data, x->data<T>(), saved_var_data, C, N, H * W * D,
                d_x->data<T>());
          }
          if (d_scale && d_bias) {
            KeBNBackwardScaleBias<
                T, block,
                framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
                d_y->data<T>(), x->data<T>(), saved_mean_data, saved_var_data,
                epsilon, N, C, H * W * D,
                d_scale->data<BatchNormParamType<T>>(),
                d_bias->data<BatchNormParamType<T>>());
          }
        }
      }

#ifdef PADDLE_WITH_HIP
// TODO(wangran16): wait for MIOpen to improve the performance of BN
// clean when exit.
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(data_desc_));
// PADDLE_ENFORCE_CUDA_SUCCESS(
//     platform::dynload::miopenDestroyTensorDescriptor(bn_param_desc_));
#else
      // clean when exit.
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(data_desc_));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnDestroyTensorDescriptor(bn_param_desc_));
#endif
    } else {
      const auto *running_mean = ctx.Input<Tensor>("Mean");
      const auto *running_var = ctx.Input<Tensor>("Variance");

      const auto *running_mean_data =
          running_mean->template data<BatchNormParamType<T>>();
      const auto *running_var_data =
          running_var->template data<BatchNormParamType<T>>();

      if (is_inplace) {
        auto px = *x;
        inplace_functor(data_layout, px.mutable_data<T>(ctx.GetPlace()),
                        scale->template data<BatchNormParamType<T>>(),
                        bias->template data<BatchNormParamType<T>>(),
                        running_mean_data, running_var_data, epsilon, C,
                        H * W * D, num, x->data<T>(), grid2, block, stream);
      }

      if (compute_format == DataLayout::kNCHW) {
        if (d_x) {
          KeBNBackwardData<
              T, framework::DataLayout::kNCHW><<<grid1, block, 0, stream>>>(
              d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
              running_var_data, epsilon, C, H * W, num, d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<
              T, block,
              framework::DataLayout::kNCHW><<<grid2, block, 0, stream>>>(
              d_y->data<T>(), x->data<T>(), running_mean_data, running_var_data,
              epsilon, N, C, H * W * D, d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      } else {
        if (d_x) {
          KeBNBackwardData<
              T, framework::DataLayout::kNHWC><<<grid1, block, 0, stream>>>(
              d_y->data<T>(), scale->data<BatchNormParamType<T>>(),
              running_var_data, epsilon, C, H * W, num, d_x->data<T>());
        }
        if (d_scale && d_bias) {
          KeBNBackwardScaleBias<
              T, block,
              framework::DataLayout::kNHWC><<<grid2, block, 0, stream>>>(
              d_y->data<T>(), x->data<T>(), running_mean_data, running_var_data,
              epsilon, N, C, H * W * D, d_scale->data<BatchNormParamType<T>>(),
              d_bias->data<BatchNormParamType<T>>());
        }
      }
    }
  }
};

template <typename T>
class BatchNormDoubleGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");

    PADDLE_ENFORCE_EQ(
        is_test, false,
        platform::errors::InvalidArgument(
            "`is_test = True` CANNOT be used in train program. If "
            "you want to use global status in pre_train model, "
            "please set `use_global_stats = True`"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");

    NormDoubleGradFunctor<platform::CUDADeviceContext, T>(
        ctx, data_layout, X, Scale, dY, Saved_mean, Saved_variance, epsilon,
        use_global_stats, ddX, ddScale, ddBias, dX, dScale, ddY);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(
    batch_norm, ops::BatchNormKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad, ops::BatchNormGradKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad_grad,
    ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(
    batch_norm, ops::BatchNormKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormKernel<plat::CUDADeviceContext, double>,
    ops::BatchNormKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad, ops::BatchNormGradKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormGradKernel<plat::CUDADeviceContext, double>,
    ops::BatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    batch_norm_grad_grad,
    ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, float>,
    ops::BatchNormDoubleGradKernel<plat::CUDADeviceContext, double>);
#endif
