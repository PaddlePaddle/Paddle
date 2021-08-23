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

#ifdef PADDLE_WITH_CUDA
#include <cuda_fp16.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_fp16.h>
#endif

#include <algorithm>
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

#ifdef __HIPCC__
constexpr int kMaxThread = 256;
constexpr int kWarpSize = 64;
#else
constexpr int kMaxThread = 128;
constexpr int kWarpSize = 32;
#endif

enum ReduceMode { GlobalMode, LocalMode };

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<platform::float16> {
 public:
  using Type = float;
};

__device__ __forceinline__ int SharedMemoryIndex(int index) {
  return (threadIdx.y + index) * blockDim.x + threadIdx.x;
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ T WarpReduce(T val, ReduceOp reducer) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int stride = details::kWarpSize / 2; stride > 0; stride >>= 1) {
    T temp = paddle::platform::CudaShuffleDownSync(mask, val, stride);
    val = reducer(val, temp);
  }
  return val;
}

/* e.g.
 * |---------block---------|
 * |warp0|warp1|warp2|warp3|
 * |0~31|32~63|64~95|96~127|  ---->blockDim.x = 128
 *  \|/  \|/   \|/    \|/     ---->1. First WarpReduce in each warp
 * res0  res1  res2  res3     ---->2. Store result of each warp to shared memory
 *   \    \    /     /        ---->3. Load the result above from shared memory
 *        res                         to warp0 and process the second WarpReduce
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockXReduce(T val, ReduceOp reducer) {
  __syncthreads();
  using details::kWarpSize;
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
  return val;
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockYReduce(T val, ReduceOp reducer) {
  __shared__ T shared_memory[details::kMaxThread];
  shared_memory[SharedMemoryIndex(0)] = val;
  for (int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.y < stride && threadIdx.y + stride < blockDim.y) {
      T temp = shared_memory[SharedMemoryIndex(stride)];
      val = reducer(val, temp);
    }
    shared_memory[SharedMemoryIndex(0)] = val;
  }
  return val;
}

}  // namespace details

/*************************** Compute Function****************************/

/**
 * @brief compute functor for elementwise_two, in1 and in2 has the same shape
 * @param：
 * T : the type of in1 and in2
 * NX: the row of in1 and in2
 * NY: the col of in1 and in2
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: ADD, SUB, XOR, OR, MUL
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(OutT* out, const T* in1,
                                                  const T* in2,
                                                  OpFunc compute) {
  T args[2];
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    args[0] = in1[idx];
    args[1] = in2[idx];
    out[idx] = static_cast<OutT>(compute(args));
  }
}

/**
 * @brief eg: a * b + c, in1 in2, in3 and out has the same shape
 * @param：
 * T : the type of in1 and in2, in3
 * NX: the row of in1, in2 and in3
 * NY: the col of in1, in2 and in3
 * BlockSize: the strid of col
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseTernary(OutT* out, const T* in1,
                                                   const T* in2, const T* in3,
                                                   OpFunc compute) {
  T args[3];
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    args[0] = in1[idx];
    args[1] = in2[idx];
    args[2] = in3[idx];
    out[idx] = static_cast<OutT>(compute(args));
  }
}

/**
 * @brief compute functor for elementwise_two, in1 is [1, NY], in2 is [NX, NY]
 * @param：
 * T : the type of in1 and in2
 * NX: the row of in1 and in2
 * NY: the col of in2
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: ADD, SUB, XOR, OR, MUL
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void CycleBinary(OutT* out, const T* in1,
                                            const T* in2, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idx + idy * NX] =
          static_cast<OutT>(compute(in1[idx], in2[idx + idy * NX]));
    }
  }
}

/**
 * @brief compute functor for unary, in1 is [NX, NY]
 * @param：
 * T : the type of in
 * NX: the row of in
 * NY: the col of in
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: relu, sigmoid, exp
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseUnary(OutT* out, const T* in,
                                                 OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute(in + idx));
  }
}

// in[NY][NX] -> in[NY]
template <typename T, int NX, int NY, int BlockSize, class OpFunc,
          int ReduceMode>
__device__ __forceinline__ void Reduce(T* out, const T* in, OpFunc reducer,
                                       bool reduce_lastDim) {
  if (ReduceMode == details::ReduceMode::GlobalMode) {
    bool block_reduce_y = (!reduce_lastDim) && (blockDim.y > 1);
    // blockYReduce
    if (block_reduce_y) {
#pragma unroll
      for (int i = 0; i < NY; i++) {
        out[i] = details::BlockYReduce<T, OpFunc>(out[i], reducer);
      }
    }

    // blockXReduce
    if (reduce_lastDim) {
#pragma unroll
      for (int i = 0; i < NY; i++) {
        out[i] = details::BlockXReduce<T, OpFunc>(out[i], reducer);
      }
    }
  } else {  // else  LocalMode
#pragma unroll
    for (int i = 0; i < NY; ++i) {
#pragma unroll
      for (int j = 0; j < NX; ++j) {
        out[i] = reducer(out[i], in[i * NX + j]);
      }
    }
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
