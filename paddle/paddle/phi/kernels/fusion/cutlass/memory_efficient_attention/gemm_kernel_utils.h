// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

//  Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
//
//  This source code is licensed under the BSD license found in the
//  LICENSE file in the root directory of this source tree.

#pragma once

#include "cutlass/arch/mma.h"
#include "paddle/phi/core/enforce.h"

////////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////////
#define DISPATCH_TYPES(tensor, func)                                         \
  {                                                                          \
    if (query.scalar_type() == at::ScalarType::Float) {                      \
      using scalar_t = float;                                                \
      func();                                                                \
    } else if (query.scalar_type() == at::ScalarType::Half) {                \
      using scalar_t = cutlass::half_t;                                      \
      func();                                                                \
    } else if (query.scalar_type() == at::ScalarType::BFloat16) {            \
      using scalar_t = cutlass::bfloat16_t;                                  \
      func();                                                                \
    } else {                                                                 \
      PADDLE_CHECK(false, "Only fp32, half & bf16 supported at the moment"); \
    }                                                                        \
  }

#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, F) \
  {                                         \
    if (BOOL_V) {                           \
      constexpr bool BOOL_NAME = true;      \
      F();                                  \
    } else {                                \
      constexpr bool BOOL_NAME = false;     \
      F();                                  \
    }                                       \
  }
#define DISPATCH_ARCHTAG(CC, func)                                        \
  {                                                                       \
    if (CC >= 80) {                                                       \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (CC >= 75) {                                                \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (CC >= 70) {                                                \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \
    } else if (CC >= 50) {                                                \
      using ArchTag = cutlass::arch::Sm50;                                \
      func();                                                             \
    } else {                                                              \
      PADDLE_CHECK(                                                       \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                          \
  PADDLE_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  PADDLE_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  PADDLE_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                      \
  PADDLE_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  PADDLE_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  PADDLE_CHECK(TENSOR.stride(-1) == 1,                                  \
               #TENSOR ": last dimension must be contiguous");

#ifdef defined(__CUDACC_RTC__)
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT)  \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) { \
    return false;                          \
  }
#define PADDLE_CHECK(COND, ERR) \
  if (!(COND)) {                \
    return false;               \
  }
#else
#include <iostream>
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT)            \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) {           \
    std::cerr << #PTR " is not correctly aligned\n"; \
    return false;                                    \
  }
#define PADDLE_CHECK(COND, ERR)     \
  if (!(COND)) {                    \
    std::cerr << #COND " failed\n"; \
    return false;                   \
  }
#endif

#define ASSIGN_CHECK_OVERFLOW(A, B)                           \
  {                                                           \
    A = B;                                                    \
    PADDLE_CHECK(B < std::numeric_limits<decltype(A)>::max(), \
                 #B " overflows");                            \
  }

namespace gemm_kernel_utils {

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer align_up(integer n, integer m) {
  return ((n + m - 1) / m) * m;
}

inline int32_t getMaximumSharedMemoryPerBlockKb(int cc) {
  // from:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
  switch (cc) {
    case 50:
    case 52:
    case 53:
    case 60:
    case 61:
    case 62:
      return 64;
    case 70:
    case 72:
      return 96;
    case 75:
      return 64;
    case 80:
      return 163;
    case 86:
      return 99;
    case 87:
      return 163;
    case 89:
      return 99;
    case 90:
      return 227;
    default:
      return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Determine the type of GEMM we do (TensorCores or not, Shapes ...)
// TODO(xformers): Maybe we could rely on Cutlass's DefaultGemm templates
////////////////////////////////////////////////////////////////////////////////

// Fallback to Simt (FMA on cuda cores) if not in a special case below
template <typename ArchTag, typename scalar_t_, typename Enable = void>
struct DefaultGemmType {
  static constexpr int ThreadK = 8;
  static constexpr int WarpK = 8;
  static constexpr int kMinimumAlignment = 1;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using OpClass = cutlass::arch::OpClassSimt;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for tensorcores with f32
template <typename ArchTag>
struct DefaultGemmType<ArchTag,
                       float,
                       typename cutlass::platform::enable_if<
                           ArchTag::kMinComputeCapability >= 80>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
};

// Specialization for tensorcores with f16/bf16 - Sm75+
template <typename ArchTag, typename scalar_t>
struct DefaultGemmType<ArchTag,
                       scalar_t,
                       typename cutlass::platform::enable_if<
                           ArchTag::kMinComputeCapability >= 75 &&
                           cutlass::sizeof_bits<scalar_t>::value == 16>::type> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 4;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Specialization for tensorcores with f16 - Volta
template <>
struct DefaultGemmType<cutlass::arch::Sm70, cutlass::half_t, void> {
  static constexpr int ThreadK = 32;
  static constexpr int WarpK = 32;
  static constexpr int kMinimumAlignment = 2;
  using OpClass = cutlass::arch::OpClassTensorOp;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// Enables to do
// `auto x = kCondition ? fa(arg) : fb(arg)`
// when `fa` and `fb` have different types
template <bool kVal, typename TA, typename TB>
struct call_conditional;

template <typename TA, typename TB>
struct call_conditional<true, TA, TB> {
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(ta(arg)) {
    return ta(arg);
  }
};

template <typename TA, typename TB>
struct call_conditional<false, TA, TB> {
  template <typename Arg>
  static CUTLASS_HOST_DEVICE auto apply(TA ta, TB tb, Arg arg)
      -> decltype(tb(arg)) {
    return tb(arg);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Mark a variable as warp-uniform - enables some compiler optimizations
// The cheapest way to do it is just to broadcast it from lane 0
////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE int32_t warp_uniform(int32_t value) {
  return (int32_t)__shfl_sync(0xffffffff, (unsigned)value, 0);
}

template <typename T>
CUTLASS_DEVICE T* warp_uniform(T* ptr) {
  struct {
    union {
      T* ptr;
      uint32_t asInt[2];
    };
  } p;
  p.ptr = ptr;
  p.asInt[0] = warp_uniform(p.asInt[0]);
  p.asInt[1] = warp_uniform(p.asInt[1]);
  return p.ptr;
}
}  // namespace gemm_kernel_utils
