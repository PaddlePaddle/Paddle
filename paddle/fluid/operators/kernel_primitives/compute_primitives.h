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

namespace paddle {
namespace operators {
namespace kernel_primitives {
/*************************** Compute Functor****************************/
template <typename T>
struct ADD {
  __device__ __forceinline__ T operator()(const T in1, const T in2) {
    return (in1 + in2);
  }
};

template <typename T>
struct RELU {
  __device__ __forceinline__ T operator()(const T in) {
    return (in > static_cast<T>(0) ? in : static_cast<T>(0.f));
  }
};

template <typename T>
struct FMA {
  __device__ __forceinline__ T operator()(const T in1, const T in2,
                                          const T in3) {
    return in1 * in2 + in3;
  }
};

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
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void elementwise_binary(const T* __restrict__ in1,
                                   const T* __restrict__ in2,
                                   T* __restrict__ out, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = compute(in1[idx], in2[idx]);
  }
}

/**
 * @brief fma eg: a * b + c, in1 in2, in3 and out has the same shape
 * @param：
 * T : the type of in1 and in2, in3
 * NX: the row of in1, in2 and in3
 * NY: the col of in1, in2 and in3
 * BlockSize: the strid of col
 */
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void elementwise_fma(const T* in1, const T* in2, const T* in3,
                                T* out, OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = compute(in1[idx], in2[idx], in3[idx]);
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
__device__ void cycle_binary(const T* in1, const T* in2, T* out,
                             OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idx + idy * NX] = compute(in1[idx], in2[idx + idy * NX]);
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
__device__ void elementwise_unary(const T* in, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX * NY; idx++) {
    out[idx] = compute(in[idx]);
  }
}

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
