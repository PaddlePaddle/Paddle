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

enum ElementwiseType { kUnary = 1, kBinary = 2 };

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
int GetVectorizedSizeImpl(const T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec2 = std::alignment_of<AlignedVector<T, 2>>::value;  // NOLINT
  if (address % vec4 == 0) {
    return 4;
  } else if (address % vec2 == 0) {
    return 2;
  }
  return 1;
}

template <typename T>
int GetVectorizedSize(const std::vector<const T *> ins,
                      const std::vector<T *> outs) {
  int vec_size = 4;
  for (auto iter = ins.begin(); iter != ins.end(); ++iter) {
    vec_size = std::min<int>(vec_size, GetVectorizedSizeImpl(*iter));
  }
  for (auto iter = outs.begin(); iter != outs.end(); ++iter) {
    vec_size = std::min<int>(vec_size, GetVectorizedSizeImpl(*iter));
  }
  return vec_size;
}

template <ElementwiseType N, int VecSize, typename T>
struct ElementwiseDataWrapper {
  T *out;
  const T *in0;
  const T *in1;
  __device__ ElementwiseDataWrapper(T *out, const T *in0,
                                    const T *in1 = nullptr)
      : out(out), in0(in0), in1(in1) {}

  using VecType = AlignedVector<T, VecSize>;

  inline __device__ void load_vector(VecType args[], int idx) {
    const VecType *x_vec = reinterpret_cast<const VecType *>(in0);
    args[0] = x_vec[idx];
    if (N == ElementwiseType::kBinary) {
      const VecType *y_vec = reinterpret_cast<const VecType *>(in1);
      args[1] = y_vec[idx];
    }
  }

  inline __device__ void load_scalar(T args[], int idx) {
    args[0] = in0[idx];
    if (N == ElementwiseType::kBinary) {
      args[1] = in1[idx];
    }
  }

  inline __device__ void store_vector(VecType res, int idx) {
    VecType *out_vec = reinterpret_cast<VecType *>(out);
    out_vec[idx] = res;
  }

  inline __device__ void store_scalar(T res, int idx) { out[idx] = res; }
};

template <ElementwiseType N, int VecSize, typename T, typename Functor>
__device__ void VectorizedKernelImpl(ElementwiseDataWrapper<N, VecSize, T> data,
                                     int size, Functor func, int tid) {
  using VecType = AlignedVector<T, VecSize>;
  const int in_num = static_cast<int>(N);
  VecType ins_vec[N];
  VecType out_vec;
  T *ins_ptr[N];
  T *out_ptr;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    ins_ptr[i] = reinterpret_cast<T *>(&(ins_vec[i]));
  }
  out_ptr = reinterpret_cast<T *>(&out_vec);

  // load
  data.load_vector(ins_vec, tid);

// compute
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    T ins[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      ins[j] = ins_ptr[j][i];
    }
    out_ptr[i] = func(ins);
  }

  // store
  data.store_vector(out_vec, tid);
}

template <ElementwiseType N, typename T, typename Functor>
__device__ void ScalarKernelImpl(ElementwiseDataWrapper<N, 1, T> data, int size,
                                 Functor func, int start, int remain) {
  const int in_num = static_cast<int>(N);
  T ins[N];
  T out;

  for (int i = 0; i < remain; ++i) {
    int idx = start + i;
    // load
    data.load_scalar(ins, idx);
    // compute
    out = func(ins);
    // store
    data.store_scalar(out, idx);
  }
}

template <ElementwiseType N, int VecSize, typename T, typename Functor>
__global__ void VectorizedKernel(const T *__restrict__ in0,
                                 const T *__restrict__ in1, T *out, int size,
                                 Functor func) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int remain = size - VecSize * tid;
  remain = remain > 0 ? remain : 0;
  if (remain >= VecSize) {
    auto data = ElementwiseDataWrapper<N, VecSize, T>(out, in0, in1);
    VectorizedKernelImpl(data, size, func, tid);
  } else {
    auto data = ElementwiseDataWrapper<N, 1, T>(out, in0, in1);
    ScalarKernelImpl(data, size, func, tid * VecSize, remain);
  }
}

template <ElementwiseType N, typename T, typename Functor>
__global__ void ScalarKernel(const T *__restrict__ in0,
                             const T *__restrict__ in1, T *out, int size,
                             Functor func) {
  auto data = ElementwiseDataWrapper<N, 1, T>(out, in0, in1);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  ScalarKernelImpl(data, size, func, tid, 1);
}

template <ElementwiseType N, typename T, typename Functor>
void LaunchElementwiseCudaKernel(const framework::ExecutionContext &ctx,
                                 const std::vector<const T *> &ins,
                                 std::vector<T *> outs, int size,
                                 Functor func) {
  // calculate the max vec_size for all ins and outs
  int vec_size = GetVectorizedSize(ins, outs);
  int block_size = PADDLE_CUDA_THREAD_SIZE;
  int grid_size =
      ((size + vec_size - 1) / vec_size + block_size - 1) / block_size;
  const T *in0 = ins[0];
  const T *in1 = nullptr;
  if (N == ElementwiseType::kBinary) {
    in1 = ins[1];
  }
  T *out = outs[0];
  // cuda kernel
  auto stream =
      ctx.template device_context<platform::CUDADeviceContext>().stream();
  switch (vec_size) {
    case 4:
      VectorizedKernel<N, 4><<<grid_size, block_size, 0, stream>>>(
          in0, in1, out, size, func);
      break;
    case 2:
      VectorizedKernel<N, 2><<<grid_size, block_size, 0, stream>>>(
          in0, in1, out, size, func);
      break;
    case 1:
      ScalarKernel<N><<<grid_size, block_size, 0, stream>>>(in0, in1, out, size,
                                                            func);
      break;
    default:
      PADDLE_THROW("Unsupported vectorized size!");
      break;
  }
}

#endif
#endif
}  // namespace operators
}  // namespace paddle
