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
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <iostream>
#include <vector>

namespace paddle {
namespace operators {
/**
 * @brief compute functor for elementwise_two, in1 and in2 has the same shape
 * @param：
 * T : the type of in1 and in2
 * NX: the row of in1 and in2
 * NY: the col of in1 and in2
 * BlockSize: the strid of col
 * OpFunc: compute functor eg: ADD, SUB, XOR, OR, MUL
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void Binary(const T* __restrict__ in1, const T* __restrict__ in2,
                       T* __restrict__ out) {
  OpFunc compute;
#pragma unroll
  for (int idy = 0; idy < NY; idy++) {
#pragma unroll
    for (int idx = 0; idx < NX; idx++) {
      compute(in1[idx + idy * NX], in2[idx + idy * NX], out[idx + idy * NX]);
    }
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
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void CycleBinary(const T* in1, const T* in2, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      compute(in1[idx], in2[idx + idy * NX], out[idx + idy * NX]);
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
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void Unary(const T* in, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idy = 0; idy < NY; idy++) {
#pragma unroll
    for (int idx = 0; idx < NX; idx++) {
      compute(in[idx + idy * NX], out[idx + idy * NX]);
    }
  }
}

/** @brief: load
 *
 */
template <typename T, int NX, int NY>
__device__ void load(const T* in, T* out, int strid_in) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idx + idy * NX] = in[idy + strid_in * idx];
    }
  }
}

/** @brief: store
 *
 */
template <typename T, int NX, int NY>
__device__ void store(const T* in, T* out, int strid_out) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idy + strid_out * idx] = in[idx + idy * NX];
    }
  }
}

// transformer_t(x)
template <typename Tx, typename Ty, int NX, int NY, typename TransformOp>
__device__ __forceinline__ void transformer_t(const Tx* in, Ty* out,
                                              TransformOp trans) {
#pragma unroll NX
#pragma unroll NY
  for (int idy = 0; idy < NY; ++idy) {
    for (int idx = 0; idx < NX; ++idx) {
      out[idy * NX + idx] = static_cast<Ty>(trans(in[idy * NX + idx]));
    }
  }
}

// reduce higher
template <typename T, int NX, int NY, typename OpFunc>
__device__ __forceinline__ void reduce(const T* in, T* out, OpFunc reducer) {
#pragma unroll NX
  for (int idx = 0; idx < NX; ++idx) {
#pragma unroll NY
    for (int idy = 0; idy < NY; ++idy) {
      out[idx] = reducer(out[idx], in[idx * NY + idy]);
    }
  }
}

// reduce lastDim
template <typename T, int NX, int NY, typename OpFunc>
__device__ __forceinline__ void reduceNX(const T* in, T* out, OpFunc reducer) {
  for (int idy = 0; idy < NY; ++idy) {
    for (int idx = 0; idx < NX; ++idx) {
      out[idy] = reducer(out[idy], in[idy * NX + idx]);
    }
  }
}
}  // namespace operators
}  // namespace paddle
