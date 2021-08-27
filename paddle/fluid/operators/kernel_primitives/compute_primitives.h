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
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

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

}  // namespace details

/*************************** Compute Functor****************************/
template <typename T, typename Enable = void>
struct DivFunctor {
  inline HOSTDEVICE T operator()(const T* args) const {
    return args[0] / args[1];
  }
};

template <typename T>
struct DivFunctor<T, typename std::enable_if_t<std::is_integral<T>::value>> {
  inline HOSTDEVICE T operator()(const T* args) const {
    PADDLE_ENFORCE(args[1] != 0,
                   platform::errors::InvalidArgument(
                       "Invalid Argument Error: Integer division by zero "
                       "encountered in divide. Please check the input value."));
    return args[0] / args[1];
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
 * @brief fma eg: a * b + c, in1 in2, in3 and out has the same shape
 * @param：
 * T : the type of in1 and in2, in3
 * NX: the row of in1, in2 and in3
 * NY: the col of in1, in2 and in3
 * BlockSize: the strid of col
 */
template <typename T, typename OutT, int NX, int NY, int BlockSize,
          class OpFunc>
__device__ __forceinline__ void ElementwiseFma(OutT* out, const T* in1,
                                               const T* in2, const T* in3,
                                               OpFunc compute) {
#pragma unroll
  for (int idx = 0; idx < NX * NY; ++idx) {
    out[idx] = static_cast<OutT>(compute(in1[idx], in2[idx], in3[idx]));
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

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
