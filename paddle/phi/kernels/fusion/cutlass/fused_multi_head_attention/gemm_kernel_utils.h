/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holdvr nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/arch/mma.h"

////////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////////
#define DISPATCH_TYPES(tensor, func)                                        \
  {                                                                         \
    if (query.scalar_type() == at::ScalarType::Float) {                     \
      using scalar_t = float;                                               \
      func();                                                               \
    } else if (query.scalar_type() == at::ScalarType::Half) {               \
      using scalar_t = cutlass::half_t;                                     \
      func();                                                               \
    } else if (query.scalar_type() == at::ScalarType::BFloat16) {           \
      using scalar_t = cutlass::bfloat16_t;                                 \
      func();                                                               \
    } else {                                                                \
      TORCH_CHECK(false, "Only fp32, half & bf16 supported at the moment"); \
    }                                                                       \
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
      TORCH_CHECK(                                                        \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                         \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(TENSOR.is_contiguous());

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                     \
  TORCH_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  TORCH_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  TORCH_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#ifdef HAS_PYTORCH
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  TORCH_CHECK(uint64_t(PTR) % ALIGNMENT == 0, #PTR " is not correctly aligned")
#define XFORMERS_CHECK TORCH_CHECK
#elif defined(__CUDACC_RTC__)
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT)  \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) { \
    return false;                          \
  }
#define XFORMERS_CHECK(COND, ERR) \
  if (!(COND)) {                  \
    return false;                 \
  }
#else
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT)            \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) {           \
    std::cerr << #PTR " is not correctly aligned\n"; \
    return false;                                    \
  }
#define XFORMERS_CHECK(COND, ERR)   \
  if (!(COND)) {                    \
    std::cerr << #COND " failed\n"; \
    return false;                   \
  }
#endif

#define ASSIGN_CHECK_OVERFLOW(A, B)                                \
  {                                                                \
    A = B;                                                         \
    TORCH_CHECK(                                                   \
        B < cutlass::platform::numeric_limits<decltype(A)>::max(), \
        #B " overflows");                                          \
  }

namespace gemm_kernel_utils {

#ifdef HAS_PYTORCH
template <typename scalar_t>
struct TypeTraits;

template <>
struct TypeTraits<cutlass::half_t> {
  using scalar_t = cutlass::half_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Half;
  }
  template <int nDim>
  static __host__ at::PackedTensorAccessor32<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return at::PackedTensorAccessor32<scalar_t, nDim>(
        (scalar_t*)(tensor.data_ptr()),
        tensor.sizes().data(),
        tensor.strides().data());
  }
};

template <>
struct TypeTraits<cutlass::bfloat16_t> {
  using scalar_t = cutlass::bfloat16_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::BFloat16;
  }
  template <int nDim>
  static __host__ at::PackedTensorAccessor32<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return at::PackedTensorAccessor32<scalar_t, nDim>(
        (scalar_t*)(tensor.data_ptr()),
        tensor.sizes().data(),
        tensor.strides().data());
  }
};

template <>
struct TypeTraits<float> {
  using scalar_t = float;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Float;
  }
  template <int nDim>
  static __host__ at::PackedTensorAccessor32<scalar_t, nDim> packed_accessor(
      at::Tensor const& tensor) {
    return tensor.packed_accessor32<scalar_t, nDim>();
  }
};
#endif

template <typename integer>
constexpr CUTLASS_HOST_DEVICE integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

////////////////////////////////////////////////////////////////////////////////
// Determine the type of GEMM we do (TensorCores or not, Shapes ...)
// TODO: Maybe we could rely on Cutlass's DefaultGemm templates
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
struct DefaultGemmType<
    ArchTag,
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
struct DefaultGemmType<
    ArchTag,
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
} // namespace gemm_kernel_utils
