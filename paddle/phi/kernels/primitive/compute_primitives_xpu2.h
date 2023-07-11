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

#include "paddle/phi/common/amp_type_traits.h"
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"
#include "xpu/kernel/simd_header.h"

namespace phi {
namespace kps {
namespace details {

// kGlobalMode: block reduce, each block gets an output;
// kLocalMode: thread reduce, each thread gets an output;
enum ReduceMode { kGlobalMode, kLocalMode };

static inline __device__ void sync_all() {
  __asm__ __volatile__(
      "sync_local\t\n"
      "csr_set csr3, %0\t\n"
      "sync_group csr3" ::"r"(-1));
}

#define ncores 64
template <typename T, typename OpFunc, int VecSize>
__device__ void BlockXReduce(T* out, const T* data, OpFunc reducer) {
  __shared__ T sum_array[ncores * VecSize];
  int core_idx = core_id() * VecSize;
  mfence();
  sync_all();

#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    mfence();
    sum_array[i * ncores + core_idx] = data[i];
    mfence();
  }
  sync_all();
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    T start = data[i * ncores];
#pragma unroll
    for (int j = 1; j < ncores; j++) {
      mfence();
      T tmp = sum_array[i * ncores + j];
      mfence();
      start = reducer(start, tmp);
      mfence();
    }
    out[i] = start;
  }
  sync_all();
}
#undef ncores

}  // namespace details

/**
 * @brief Perform unary calculation according to OpFunc. Shape of input and
 * output are the same.
 *
 * @template paraments
 * InT: The data type of in.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const InT& a) const {
 *         return ...;
 *       }
 *     };
 *
 * @param：
 * out: The register pointer of out, the size is NX * NY.
 * in: The register pointer of in, the size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT, OutT>().
 */
template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void ElementwiseUnary(OutT* out,
                                                 const InT* in,
                                                 OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute(in[idx]));
  }
}

/**
 * @brief Binary calculation according to OpFunc. Shape of The input and output
 * are the same.
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns computed by each thread.
 * NY: The number of data rows computed by each thread.
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 *
 * @param：
 * out: The register pointer of out, the size is NX * NY.
 * in1: The register pointer of fist input, size is NX * NY.
 * in2: The register pointer of second input, size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(OutT* out,
                                                  const InT* in1,
                                                  const InT* in2,
                                                  OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx]));
  }
}

template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(
    OutT* out, const InT* in1, const InT* in2, OpFunc compute, int read_lens) {
  for (int idx = 0; idx < read_lens; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx]));
  }
}

/**
 * @brief Ternary calculation according to OpFunc. Shape of input and output
 * are the same.
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b, const InT& c)
 * const {
 *         return ...;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in1: The register pointer of fist input, size is NX * NY.
 * in2: The register pointer of second input, size is NX * NY.
 * in3: The register pointer of third input, size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void ElementwiseTernary(
    OutT* out, const InT* in1, const InT* in2, const InT* in3, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx], in3[idx]));
  }
}

/**
 * @brief Multivariate calculation according to OpFunc. Shape of inputs and
 * output are the same.
 *
 * @template paraments
 * InT: The data type of in1, in2 and in3.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * core_id() is used as the index.
 * Arity: The size of ins
 * OpFunc: Compute functor which has an operator() as following:
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()(const InT* args) const {
 *         return ...;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * ins: A pointers of array consisting of multiple inputs.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT, typename OutT, int NX, int NY, int Arity, class OpFunc>
__device__ __forceinline__ void ElementwiseAny(OutT* out,
                                               InT (*ins)[NX * NY],
                                               OpFunc compute) {
  __local__ InT args[Arity];
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
 * @brief Binary calculation according to OpFunc. The shape of in1 and in2 are
 * different. When in1's shape is [1, NX], in2's shape is [NY, NX], then
 * output's shape is [NY, NX].
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following
 *     template <typename InT, typename OutT>
 *     struct XxxFunctor {
 *       HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in1: The register pointer of fist input, size is NX * 1.
 * in2: The register pointer of second input, size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT, OutT>().
 */
template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void CycleBinary(OutT* out,
                                            const InT* in1,
                                            const InT* in2,
                                            OpFunc compute) {
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
 * @brief The Reduce provides collective methods for computing a parallel
 * reduction of items partitioned across a CUDA block and intra thread. When
 * ReduceMode == kLocalMode, thread reduce along nx. When ReduceMode ==
 * kGlobalMode, use shared memory to reduce between threads.
 *
 * @template paraments
 * T: The type of data.
 * NX: The number of data continuously loaded by each thread.
 * NY: The number of data rows loaded by each thread, only NY = 1 was supported.
 * core_id() is used as the index.
 * ReduceFunctor: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct ReduceFunctor {
 *       HOSTDEVICE InT operator()(const InT& a, const InT& b) const {
 *         return ...;
 *       }
 *     };
 * ReduceMode: Reduce mode, can be kLocalMode, kGlobalMode.
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * in: The register pointer of in, the size is NX * NY.
 * reducer: Compute function which was declared like ReduceFunctor<InT>().
 * reduce_last_dim: if the last dim gets involved in reduction.
 */
template <typename T,
          int NX,
          int NY,
          class ReduceFunctor,
          details::ReduceMode Mode>
__device__ __forceinline__ void Reduce(T* out,
                                       const T* in,
                                       ReduceFunctor reducer,
                                       bool reduce_last_dim) {
  if (Mode == details::kGlobalMode) {
    if (reduce_last_dim) {
#pragma unroll
      for (int i = 0; i < NY * NX; i++) {  // reduce along blockDim.x
        details::BlockXReduce<T, ReduceFunctor, 1>(&out[i], &in[i], reducer);
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

/*
 * @brief Fill register with a constant according to OpFunc
 *
 * @template paraments
 * InT: The data type of in1 and in2.
 * OutT: The data type of out.
 * NX: The number of data columns loaded by each thread.
 * NY: The number of data rows loaded by each thread.
 * core_id() is used as the index.
 * OpFunc: Compute functor which has an operator() as following
 *     template <typename InT>
 *     struct XxxFunctor {
 *       HOSTDEVICE InT operator()()
 * const {
 *         return a;
 *       }
 *     };
 *
 * @param
 * out: The register pointer of out, the size is NX * NY.
 * compute: Compute function which was declared like OpFunc<InT>().
 */
template <typename InT, typename OutT, int NX, int NY, class OpFunc>
__device__ __forceinline__ void ElementwiseConstant(OutT* out, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = static_cast<OutT>(compute());
  }
}

}  // namespace kps
}  // namespace phi
