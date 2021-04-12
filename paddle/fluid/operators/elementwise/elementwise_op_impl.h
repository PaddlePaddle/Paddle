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

enum ELEMENTWISE_TYPE {
  UNARY = 1,
  BINARY = 2,
};

template <typename T>
inline int SameDimsVectorizedSize(const T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  int vec_size = sizeof(float4) / sizeof(T);
  if (address % sizeof(float4) == 0) {
    return vec_size;
  }
  return 1;
}

template <typename T, ELEMENTWISE_TYPE N>
struct SameDimsData {
  using VecType = float4;
  T *out = nullptr;
  const T *in0 = nullptr;
  const T *in1 = nullptr;
  SameDimsData(T *out, const T *in0, const T *in1 = nullptr)
      : out(out), in0(in0), in1(in1) {}

  void get_vectorized_size(int *vec_size) const {
    *vec_size = std::min<int>(*vec_size, SameDimsVectorizedSize<T>(out));
    *vec_size = std::min<int>(*vec_size, SameDimsVectorizedSize<T>(in0));
    if (N == BINARY) {
      *vec_size = std::min<int>(*vec_size, SameDimsVectorizedSize<T>(in1));
    }
  }

  inline __device__ void load_vector(VecType args[], int tid) const {
    const VecType *x_vec = reinterpret_cast<const VecType *>(in0);
    args[1] = x_vec[tid];
    if (N == BINARY) {
      const VecType *y_vec = reinterpret_cast<const VecType *>(in1);
      args[2] = y_vec[tid];
    }
  }

  inline __device__ void load_scalar(T args[], int tid) const {
    args[1] = in0[tid];
    if (N == BINARY) {
      args[2] = in1[tid];
    }
  }

  inline __device__ void store_vector(VecType res, int tid) const {
    VecType *out_vec = reinterpret_cast<VecType *>(out);
    out_vec[tid] = res;
  }

  inline __device__ void store_scalar(T res, int tid) const { out[tid] = res; }
};

template <int Vec_size, typename T, typename Functor, ELEMENTWISE_TYPE N>
__device__ void VectorizedKernelHelper(const SameDimsData<T, N> &data, int size,
                                       Functor func, int tid) {
  using VecType = float4;
  VecType args_vec[3];
  T *args_ptr[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    args_ptr[i] = reinterpret_cast<T *>(&(args_vec[i]));
  }

  // load
  data.load_vector(args_vec, tid);

// compute
#pragma unroll
  for (int i = 0; i < Vec_size; ++i) {
    T args[3] = {args_ptr[0][i], args_ptr[1][i], args_ptr[2][i]};
    func(args);
  }

  // store
  data.store_vector(args_vec[0], tid);
}

template <typename T, typename Functor, ELEMENTWISE_TYPE N>
__device__ void ScalarKernelHelper(const SameDimsData<T, N> &data, int size,
                                   Functor func, int start, int remain) {
  T args[3];

  for (int i = 0; i < remain; ++i) {
    int tid = start + i;
    // load
    data.load_scalar(args, tid);
    // compute
    func(args);
    // store
    data.store_scalar(args[0], tid);
  }
}

template <int Vec_size, typename T, typename Functor, ELEMENTWISE_TYPE N>
__global__ void VectorizedSameDimsKernel(const SameDimsData<T, N> &data,
                                         int size, Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int remain = size - Vec_size * tid;
  remain = remain > 0 ? remain : 0;
  if (remain >= Vec_size) {
    VectorizedKernelHelper<Vec_size>(data, size, func, tid);
  } else {
    ScalarKernelHelper(data, size, func, tid * Vec_size, remain);
  }
}

template <typename T, typename Functor, ELEMENTWISE_TYPE N>
__global__ void ScalarSameDimsKernel(const SameDimsData<T, N> &data, int size,
                                     Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ScalarKernelHelper(data, size, func, tid, 1);
}

template <typename T, typename Functor, ELEMENTWISE_TYPE N>
void same_dims_launch_kernel(const framework::ExecutionContext &ctx,
                             const SameDimsData<T, N> &data, int64_t size,
                             Functor func) {
  // calculate the max vec_size for all inputs and outputs
  int vec_size = 1;
  data.get_vectorized_size(&vec_size);
  int block_size = PADDLE_CUDA_THREAD_SIZE;
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  printf("===============\n");
  printf("%d\n", N);
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
