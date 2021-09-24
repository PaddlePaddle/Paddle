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

#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

#ifdef __HIPCC__
constexpr int kReduceMaxThread = 256;
constexpr int kWarpSize = 64;
#else
constexpr int kReduceMaxThread = 128;
constexpr int kWarpSize = 32;
#endif

// kGlobalMode: block reduce, each block gets an output;
// kLocalMode: thread reduce, each thread gets an output;
enum ReduceMode { kGlobalMode, kLocalMode };

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

/**
 * @brief will be used in BlockYReduce, get the index of reduce_num in shared
 * memory
 */
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

/**
 * @brief BlockXReduce reduce along blockDim.x
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

/**
 * @brief BlockYReduce reduce along blockDim.y
 */
template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockYReduce(T val, ReduceOp reducer) {
  __shared__ T shared_memory[details::kReduceMaxThread];
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

/**
 * @brief unary function
 * @param
 * T: data type of in
 * OutT: data type of out
 * NX: the cols of in
 * NY: the rows of in
 * BlockSize: the config of this device
 * OpFunc: compute functor which have an operator() as following
 *     template <typename T, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const T& a) const {
 *         return ...;
 *       }
 *     };
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseUnary(OutT* out, const T* in,
                                                 OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute(in[idx]));
  }
}

/**
 * @brief binary function, in1 and in2 have same shape
 * @param
 * T: data type of in1, in2
 * OutT: data type of out
 * NX: the cols of in1, in2
 * NY: the rows of in1, in2
 * BlockSize: the config of this device
 * OpFunc: compute functor which have an operator() as following
 *     template <typename T, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const T& a, const T& b) const {
 *         return ...;
 *       }
 *     };
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(OutT* out, const T* in1,
                                                  const T* in2,
                                                  OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx]));
  }
}

/**
 * @brief ternary function, in1, in2 and in3 have same shape
 * @param
 * T: data type of in1, in2, in3
 * OutT: data type of out
 * NX: the cols of in1, in2
 * NY: the rows of in1, in2
 * BlockSize: the config of this device
 * OpFunc: compute functor which have an operator() as following
 *     template <typename T, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const T& a, const T& b, const T& c) const {
 *         return ...;
 *       }
 *     };
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseTernary(OutT* out, const T* in1,
                                                   const T* in2, const T* in3,
                                                   OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx], in3[idx]));
  }
}

/**
 * @brief a general function for elementwise computation, all inputs have
 *        the same shape.
 * @param
 * T: data type of in1, in2, in3
 * OutT: data type of out
 * NX: the cols of in1, in2
 * NY: the rows of in1, in2
 * BlockSize: the config of this device
 * OpFunc: compute functor which have an operator() as following
 *     template <typename T, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const T* args) const {
 *         return ...;
 *       }
 *     };
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize, int Arity,
          class OpFunc>
__device__ __forceinline__ void ElementwiseAny(OutT* out, T (*ins)[NX * NY],
                                               OpFunc compute) {
  T args[Arity];
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
#pragma unroll
    for (int j = 0; j < Arity; ++j) {
      args[j] = ins[j][idx];
    }
    out[idx] = static_cast<OutT>(compute(args));
  }
}

/**
 * @brief cycle binary function, in1's shape size is [1, NX], in2's shape size
 * is [NY, NX], out's shape size is [NY, NX]
 * @param
 * T: data type of in1, in2
 * OutT: data type of out
 * NX: the cols of in1, in2
 * NY: the rows of in1, in2
 * BlockSize: the config of this device
 * OpFunc: compute functor eg: in1 + in2, in1 - in2
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
 * @brief reduce function, in's shape size is [NX, NY].
 * If ReduceMode == kLocalMode then reduce NX, the shape of out is [NY, 1],
 * if ReduceMode == kGlobalMode then reduce between different threads, the
 * shape of out is [NY, NX]. If reduce_last_dim is false and reduce_num was
 * split, BlockYReduce will be called. If reduce_last_dim is true and
 * reduce_num was split, BlockXReduce will be called
 * @typename
 * T: data type of in
 * NX: the cols of in
 * NY: the rows of in
 * BlockSize: the config of this device
 * OpFunc: reduce functor, eg: CustomSum, CustomMean in reduce_functor_op.h
 * @param:
 * reducer: reduce functor, eg: CustomSum<T>()
 * reduce_last_dim: if in's last dim need to be reduce then reduce_last_dim =
 * true
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc,
          details::ReduceMode Mode>
__device__ __forceinline__ void Reduce(T* out, const T* in, OpFunc reducer,
                                       bool reduce_last_dim) {
  int block_index = blockDim.y;

  if (Mode == details::ReduceMode::kGlobalMode) {
    bool block_reduce_y = (!reduce_last_dim) && (block_index > 1);
    // when reduce is not required for the last dim, and reduce num has been
    // split into multiple threads
    if (block_reduce_y) {
#pragma unroll
      for (int i = 0; i < NY * NX; i++) {  // reduce along blockdim.y
        out[i] = details::BlockYReduce<T, OpFunc>(out[i], reducer);
      }
    }

    // when last dimension need to be reduced
    if (reduce_last_dim) {
#pragma unroll
      for (int i = 0; i < NY * NX; i++) {  // reduce along blockDim.x
        out[i] = details::BlockXReduce<T, OpFunc>(out[i], reducer);
      }
    }
  } else {  // else  kLocalMode
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
