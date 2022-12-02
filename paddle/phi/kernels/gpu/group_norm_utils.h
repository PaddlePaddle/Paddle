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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

enum GroupNormKernelFlags { kHasScale = 1, kHasBias = 2 };
#define ALIGN_BYTES 16

#define CHECK_CASE(i, flags, kernel_name, ...)                 \
  if (i == flags) {                                            \
    kernel_name<T, AccT, i>                                    \
        <<<grid, threads, 0, dev_ctx.stream()>>>(__VA_ARGS__); \
  }

// 0 for no scale, no bias
// 1 for has scale, no bias
// 2 for no scale, has bias
// 3 for has scale, has bias
#define UNROLL_ALL_CASES(flags, kernel_name, ...) \
  CHECK_CASE(0, flags, kernel_name, __VA_ARGS__)  \
  CHECK_CASE(1, flags, kernel_name, __VA_ARGS__)  \
  CHECK_CASE(2, flags, kernel_name, __VA_ARGS__)  \
  CHECK_CASE(3, flags, kernel_name, __VA_ARGS__)

template <typename T>
__device__ __inline__ void CudaAtomicAddWithWarp(T* sum, T value) {
  typedef cub::WarpReduce<T> WarpReduce;
  typename WarpReduce::TempStorage temp_storage;
  value = WarpReduce(temp_storage).Sum(value);
  if (cub::LaneId() == 0) phi::CudaAtomicAdd(sum, value);
}

template <typename T, typename AccT, int VecSize, int Num>
__device__ __forceinline__ void ThreadReduce(phi::Array<const T*, Num> arrs,
                                             int size,
                                             const int offset,
                                             AccT* out_mean,
                                             AccT* out_var) {
  const T* x = arrs[0];
  const T* y;
  if (Num == 2) {
    y = arrs[1];
  }
  using VecT = kps::details::VectorType<T, VecSize>;
  int tid = threadIdx.x;
  if (offset > 0) {
    x -= offset;
    if (Num == 2) {
      y -= offset;
    }
    size += offset;
    if (tid >= offset) {
      if (Num == 1) {
        AccT x_acc = static_cast<AccT>(x[tid]);
        *out_mean += x_acc;
        *out_var += x_acc * x_acc;
      } else if (Num == 2) {
        AccT x_acc = static_cast<AccT>(x[tid]);
        AccT y_acc = static_cast<AccT>(y[tid]);
        *out_mean += y_acc;
        *out_var += y_acc * x_acc;
      }
    }
    size -= blockDim.x;
    x += blockDim.x;
    if (Num == 2) {
      y += blockDim.x;
    }
  }
  int remain = size % (VecSize * blockDim.x);

  T ins_x[VecSize];
  T ins_y[VecSize];
  VecT* ins_vec_x = reinterpret_cast<VecT*>(&ins_x);
  VecT* ins_vec_y = reinterpret_cast<VecT*>(&ins_y);

  // vector part
  for (; VecSize * tid < (size - remain); tid += blockDim.x) {
    *ins_vec_x = reinterpret_cast<const VecT*>(x)[tid];
    if (Num == 2) {
      *ins_vec_y = reinterpret_cast<const VecT*>(y)[tid];
    }

#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      if (Num == 1) {
        AccT ins_x_acc = static_cast<AccT>(ins_x[i]);
        *out_mean += ins_x_acc;
        *out_var += ins_x_acc * ins_x_acc;
      } else if (Num == 2) {
        AccT ins_x_acc = static_cast<AccT>(ins_x[i]);
        AccT ins_y_acc = static_cast<AccT>(ins_y[i]);
        *out_mean += ins_y_acc;
        *out_var += ins_y_acc * ins_x_acc;
      }
    }
  }

  // scalar part
  tid = size - remain + threadIdx.x;
  for (; tid < size; tid += blockDim.x) {
    if (Num == 1) {
      AccT x_acc = static_cast<AccT>(x[tid]);
      *out_mean += x_acc;
      *out_var += x_acc * x_acc;
    } else if (Num == 2) {
      AccT x_acc = static_cast<AccT>(x[tid]);
      AccT y_acc = static_cast<AccT>(y[tid]);
      *out_mean += y_acc;
      *out_var += y_acc * x_acc;
    }
  }
}

template <typename T>
__device__ __forceinline__ void ReduceMeanAndVar(
    T* mean, T* var, T x_mean, T x_var, int size) {
  const int nc = blockIdx.x;
  x_mean = kps::details::BlockXReduce<T, kps::AddFunctor<T>>(
      x_mean, kps::AddFunctor<T>());
  x_var = kps::details::BlockXReduce<T, kps::AddFunctor<T>>(
      x_var, kps::AddFunctor<T>());
  __syncthreads();
  if (threadIdx.x == 0) {
    mean[nc] = x_mean / size;
    var[nc] = x_var / size;
  }
}

template <typename T, typename AccT>
__global__ void ScalarGetMeanAndVarNCHW(const T* x,
                                        AccT* mean,
                                        AccT* var,
                                        int size) {
  int i = blockIdx.x;
  AccT x_mean = static_cast<AccT>(0);
  AccT x_var = static_cast<AccT>(0);
  for (int j = threadIdx.x; j < size; j += blockDim.x) {
    AccT val;
    val = static_cast<AccT>(x[i * size + j]);
    x_mean += val;
    x_var += val * val;
  }
  ReduceMeanAndVar<AccT>(mean, var, x_mean, x_var, size);
}

template <typename T, typename AccT, int VecSize>
__global__ void VectorizedGetMeanAndVarNCHW(const T* x,
                                            AccT* mean,
                                            AccT* var,
                                            int size) {
  int i = blockIdx.x;
  AccT x_mean = static_cast<AccT>(0);
  AccT x_var = static_cast<AccT>(0);
  x += i * size;
  const int input_offset = ((uint64_t)x) % ALIGN_BYTES / sizeof(T);
  phi::Array<const T*, 1> ins;
  ins[0] = x;
  ThreadReduce<T, AccT, VecSize, 1>(ins, size, input_offset, &x_mean, &x_var);
  ReduceMeanAndVar<AccT>(mean, var, x_mean, x_var, size);
}

}  // namespace phi
