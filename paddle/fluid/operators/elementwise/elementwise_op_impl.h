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
#pragma once

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_fp16.h>
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#endif

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_CUDA
#ifdef __NVCC__

template <typename T>
inline int SameDimsVectorizedSize(const T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  int vec_size = sizeof(float4) / sizeof(T);
  if (address % sizeof(float4) == 0) {
    return vec_size;
  }
  return 1;
}

template <typename T>
struct SameDimsData {
  T *out = nullptr;
  const T *in0 = nullptr;
  const T *in1 = nullptr;
  int data_num = 3;
  bool is_binary = true;
  SameDimsData(T *out, const T *in0, const T *in1 = nullptr)
      : out(out), in0(in0), in1(in1) {
    if (in1 == nullptr) {
      is_binary = false;
      data_num--;
    }
  }

  int GetVectorizedSize() {
    int vec_size = 8;
    vec_size = std::min<int>(vec_size, SameDimsVectorizedSize<T>(out));
    vec_size = std::min<int>(vec_size, SameDimsVectorizedSize<T>(in0));
    if (in1 != nullptr) {
      vec_size = std::min<int>(vec_size, SameDimsVectorizedSize<T>(in1));
    }
    return vec_size;
  }
};

template <int Vec_size, typename T, typename Functor>
__device__ void VectorizedKernelHelper(SameDimsData<T> data, int size,
                                       Functor func, int tid) {
  using VecType = float4;
  const VecType *x = reinterpret_cast<const VecType *>(data.in0);
  const VecType *y = reinterpret_cast<const VecType *>(data.in1);
  VecType *z = reinterpret_cast<VecType *>(data.out);
  VecType x_vec, y_vec, z_vec;

  T *x_slr = reinterpret_cast<T *>(&x_vec);
  T *y_slr = reinterpret_cast<T *>(&y_vec);
  T *z_slr = reinterpret_cast<T *>(&z_vec);

  x_vec = x[tid];
  y_vec = y[tid];

#pragma unroll
  for (int i = 0; i < Vec_size; ++i) {
    z_slr[i] = func(x_slr[i], y_slr[i]);
  }

  z[tid] = z_vec;
}

template <typename T, typename Functor>
__device__ void ScalarKernelHelper(SameDimsData<T> data, int size, Functor func,
                                   int start, int remain) {
  for (int i = 0; i < remain; ++i) {
    T x = (data.in0)[start + i];
    T y = (data.in1)[start + i];
    (data.out)[start + i] = func(x, y);
  }
}

template <int Vec_size, typename T, typename Functor>
__global__ void VectorizedSameDimsKernel(SameDimsData<T> data, int size,
                                         Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int remain = size - Vec_size * tid;
  remain = remain > 0 ? remain : 0;
  if (remain >= Vec_size) {
    VectorizedKernelHelper<Vec_size>(data, size, func, tid);
  } else {
    ScalarKernelHelper(data, size, func, tid * Vec_size, remain);
  }
}

template <typename T, typename Functor>
__global__ void ScalarSameDimsKernel(SameDimsData<T> data, int size,
                                     Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ScalarKernelHelper(data, size, func, tid, 1);
}

template <typename T, typename Functor>
void same_dims_launch_kernel(const framework::ExecutionContext &ctx,
                             SameDimsData<T> data, int64_t size, Functor func) {
  // calculate the max vec_size for all inputs and outputs
  int vec_size = data.GetVectorizedSize();
  int block_size = PADDLE_CUDA_THREAD_SIZE;
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  printf("===============\n");
  printf("size: %d\n", size);
  printf("vec_size: %d\n", vec_size);
  printf("block_size: %d\n", block_size);
  printf("grid_size: %d\n", grid_size);
  printf("===============\n");
  // cuda kernel
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  switch (vec_size) {
    case 8:
      VectorizedSameDimsKernel<8><<<grid_size, block_size, 0, stream>>>(
          data, size, func);
      break;
    case 4:
      VectorizedSameDimsKernel<4><<<grid_size, block_size, 0, stream>>>(
          data, size, func);
      break;
    case 2:
      VectorizedSameDimsKernel<2><<<grid_size, block_size, 0, stream>>>(
          data, size, func);
      break;
    case 1:
      ScalarSameDimsKernel<<<grid_size, block_size, 0, stream>>>(data, size,
                                                                 func);
      break;
    default:
      VLOG(3) << "Unsupported vectorized size!";
  }
}

#endif
#endif
}  // namespace operators
}  // namespace paddle
