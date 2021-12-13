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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#ifdef __HIPCC__
#define LAUNCH_BOUNDS(BlockDim) __launch_bounds__(BlockDim)
#else
#define LAUNCH_BOUNDS(BlockDim)
#endif

#include "paddle/fluid/operators/elementwise/elementwise_functor.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/kernel_primitives/kernel_primitives.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/platform/fast_divmod.h"

namespace paddle {
namespace operators {

#define MAX_INPUT_NUM 2

namespace kps = paddle::operators::kernel_primitives;

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using ReduceParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename InT, typename OutT, int ShapeSize, int VecSize,
          int DATA_PER_THREAD, typename Functor>
__global__ void BroadcastKernelBinary(
    const InT* __restrict__ in0, const InT* __restrict__ in1, OutT* out,
    framework::Array<bool, MAX_INPUT_NUM> use_broadcast, uint32_t numel,
    framework::Array<kps::details::BroadcastConfig<ShapeSize>, MAX_INPUT_NUM>
        configlists,
    int main_tid, int tail_tid, Functor func) {
  int fix = blockIdx.x * blockDim.x * VecSize;
  int num = tail_tid;
  InT arg0[VecSize * DATA_PER_THREAD];
  InT arg1[VecSize * DATA_PER_THREAD];
  OutT result[VecSize * DATA_PER_THREAD];
  if (blockIdx.x < main_tid) {
    num = blockDim.x * VecSize;  // blockIdx.x < main_tid
  }

  // load in0
  if (use_broadcast[0]) {
    kernel_primitives::ReadDataBc<InT, VecSize, DATA_PER_THREAD, 1, ShapeSize>(
        arg0, in0, fix, configlists[0], numel);
  } else {
    kernel_primitives::ReadData<InT, VecSize, 1, 1>(arg0, in0 + fix, num);
  }
  // load in1
  if (use_broadcast[1]) {
    kernel_primitives::ReadDataBc<InT, VecSize, DATA_PER_THREAD, 1, ShapeSize>(
        arg1, in1, fix, configlists[1], numel);
  } else {
    kernel_primitives::ReadData<InT, VecSize, 1, 1>(arg1, in1 + fix, num);
  }
  // compute
  kernel_primitives::ElementwiseBinary<InT, OutT, VecSize, 1, 1, Functor>(
      result, arg0, arg1, func);
  // store
  kernel_primitives::WriteData<OutT, VecSize, 1, 1, true>(out + fix, result,
                                                          num);
}

// bias add forward impl for "[m, n] + [n] = [m, n]"
template <typename T>
void LaunchBiasAddFwKernel(const platform::CUDADeviceContext& ctx, int m, int n,
                           const T* in0, const T* in1, T* out) {
  int in_vec_size = std::min(platform::GetVectorizedSize<T>(in0),
                             platform::GetVectorizedSize<T>(in1));
  int out_vec_size = std::min(4, platform::GetVectorizedSize<T>(out));
  int vec_size = std::min(out_vec_size, in_vec_size);

  int numel = m * n;
  const int threads = 256;
  const int data_per_thread = 1;
  int blocks =
      ((numel + vec_size * data_per_thread - 1) / (vec_size * data_per_thread) +
       threads - 1) /
      threads;
  int main_tid = numel / (data_per_thread * vec_size * threads);
  int tail_tid = numel % (data_per_thread * vec_size * threads);

  framework::Array<kps::details::BroadcastConfig<2>, MAX_INPUT_NUM> configlists;
  framework::Array<bool, MAX_INPUT_NUM> use_broadcast;

  use_broadcast[0] = false;
  use_broadcast[1] = false;
  if (m != 1) {
    use_broadcast[1] = true;
  }
  // Here, dims are transposed due to the logic in BroadcastConfig.
  std::vector<int64_t> input1_dims = {n, 1};
  std::vector<int64_t> out_dims = {n, m};
  configlists[1] = kps::details::BroadcastConfig<2>(out_dims, input1_dims, 2);

  auto func = AddFunctor<T>();
  auto stream = ctx.stream();
  switch (vec_size) {
    case 4: {
      BroadcastKernelBinary<T, T, 2, 4,
                            data_per_thread><<<blocks, threads, 0, stream>>>(
          in0, in1, out, use_broadcast, numel, configlists, main_tid, tail_tid,
          func);
      break;
    }
    case 2: {
      BroadcastKernelBinary<T, T, 2, 2,
                            data_per_thread><<<blocks, threads, 0, stream>>>(
          in0, in1, out, use_broadcast, numel, configlists, main_tid, tail_tid,
          func);
      break;
    }
    case 1: {
      BroadcastKernelBinary<T, T, 2, 1,
                            data_per_thread><<<blocks, threads, 0, stream>>>(
          in0, in1, out, use_broadcast, numel, configlists, main_tid, tail_tid,
          func);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported vectorized size: %d !", vec_size));
      break;
    }
  }
}

template <typename T, int BlockDim>
__global__ void LAUNCH_BOUNDS(BlockDim)
    Compute1DColumnReduceKernel(const int reduce_num, const int left_num,
                                const T* in, T* out) {
  typedef cub::BlockReduce<ReduceParamType<T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage mean_storage;

  for (int i = blockIdx.x; i < left_num; i += gridDim.x) {
    ReduceParamType<T> x_sum = static_cast<ReduceParamType<T>>(0);
    for (int j = threadIdx.x; j < reduce_num; j += blockDim.x) {
      const int index = j * left_num + i;
      ReduceParamType<T> x_i = static_cast<ReduceParamType<T>>(in[index]);
      x_sum += x_i;
    }
    x_sum = BlockReduce(mean_storage).Reduce(x_sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[i] = static_cast<T>(x_sum);
    }
  }
}

template <typename T>
void Launch1DColumnReduce(gpuStream_t stream, const int max_threads,
                          const int reduce_num, const int left_num,
                          const T* d_out, T* d_bias) {
  const int block = 256;
  const int max_blocks = std::max(max_threads / block, 1);
  const int grid = std::min(left_num, max_blocks);
  Compute1DColumnReduceKernel<T, block><<<grid, block, 0, stream>>>(
      reduce_num, left_num, d_out, d_bias);
}

void SetConfigForColumnReduce(const int max_threads, const int reduce_num,
                              const int left_num, int* blocking_size,
                              bool* should_reduce_again, dim3* block_dim,
                              dim3* grid_dim) {
  block_dim->z = 1;
  grid_dim->z = 1;
  *should_reduce_again = false;

  int num_block = (max_threads / left_num);
  if (num_block > 1 && reduce_num >= REDUCE_SPLIT_BOUNDARY) {
    *blocking_size = details::GetLastPow2(reduce_num / num_block);
    if (*blocking_size <= 1) {
      *blocking_size = details::GetLastPow2(sqrt(reduce_num));
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
__global__ void BiasAddBwSinglePassKernel(const T* in, int reduce_num,
                                          int left_num, T* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ReduceParamType<T> x_sum = static_cast<ReduceParamType<T>>(0);
  if (idx < left_num) {
    for (int iy = 0; iy < reduce_num; iy++) {
      int id = iy * left_num + idx;
      ReduceParamType<T> x_val = static_cast<ReduceParamType<T>>(in[id]);
      x_sum += x_val;
    }
    out[idx] = static_cast<T>(x_sum);
  }
}

template <typename T>
__global__ void BiasAddBw2DReduceKernel(const T* x, int reduce_num,
                                        int left_num, int workload_per_thread,
                                        ReduceParamType<T>* temp_x_sum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * workload_per_thread;

  T x_val;
  ReduceParamType<T> x_sum = static_cast<ReduceParamType<T>>(0);
  if (idx < left_num) {
    int loop = reduce_num - idy;
    loop = loop > workload_per_thread ? workload_per_thread : loop;
    for (int iy = 0; iy < loop; iy++) {
      int id = (idy + iy) * left_num + idx;
      ReduceParamType<T> x_val = static_cast<ReduceParamType<T>>(x[id]);
      x_sum += x_val;
    }
    temp_x_sum[idx + blockIdx.y * left_num] = x_sum;
  }
}

template <typename T>
__global__ void BiasAddBw1DReduceKernel(const ReduceParamType<T>* temp_sum,
                                        int workload_per_thread, int left_num,
                                        T* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ReduceParamType<T> x_sum = static_cast<ReduceParamType<T>>(0);
  if (idx < left_num) {
    for (int iy = 0; iy < workload_per_thread; iy++) {
      int id = iy * left_num + idx;
      x_sum += temp_sum[id];
    }
    out[idx] = static_cast<T>(x_sum);
  }
}

template <typename T>
void Launch2DColumnReduce(const platform::CUDADeviceContext& dev_ctx,
                          const int max_threads, const int reduce_num,
                          const int left_num, const T* d_out, T* d_bias) {
  dim3 block;
  dim3 grid;
  bool should_reduce_again = false;
  int blocking_size = 1;
  SetConfigForColumnReduce(max_threads, reduce_num, left_num, &blocking_size,
                           &should_reduce_again, &block, &grid);
  const auto& stream = dev_ctx.stream();

  if (!should_reduce_again) {
    BiasAddBwSinglePassKernel<T><<<grid, block, 0, stream>>>(d_out, reduce_num,
                                                             left_num, d_bias);
  } else {
    framework::Tensor tmp_sum;
    tmp_sum.Resize({grid.y, left_num});
    tmp_sum.mutable_data<ReduceParamType<T>>(dev_ctx.GetPlace());

    BiasAddBw2DReduceKernel<T><<<grid, block, 0, stream>>>(
        d_out, reduce_num, left_num, blocking_size,
        tmp_sum.template data<ReduceParamType<T>>());

    BiasAddBw1DReduceKernel<T><<<grid.x, block.x, 0, stream>>>(
        tmp_sum.template data<ReduceParamType<T>>(), grid.y, left_num, d_bias);
  }
}

// bias add backward impl whose pattern are column-reduce with d_out[m, n] as
// input
// and d_bias[n] as output.
template <typename T>
void LaunchBiasAddBwKernel(const platform::CUDADeviceContext& dev_ctx, int m,
                           int n, const T* d_out, T* d_bias) {
  int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  int reduce_num = m;
  int left_num = n;
  bool is_large_enough = (reduce_num > REDUCE_SPLIT_BOUNDARY / 2) ||
                         (left_num > REDUCE_SPLIT_BOUNDARY);
  if (!is_large_enough) {
    Launch1DColumnReduce(dev_ctx.stream(), max_threads, reduce_num, left_num,
                         d_out, d_bias);
  } else {
    Launch2DColumnReduce(dev_ctx, max_threads, reduce_num, left_num, d_out,
                         d_bias);
  }
}

#undef MAX_INPUT_NUM

}  // namespace operators
}  // namespace paddle
