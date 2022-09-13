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
 * 3. Neither the name of the copyright holder nor the names of its
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

/*! \file
    \brief Basic include for CUTLASS.
*/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_NAMESPACE
#define cutlass CUTLASS_NAMESPACE
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_UNUSED(expr) do { (void)(expr); } while (0)

#if !defined(__CUDACC_RTC__)

#include <assert.h>

#if defined(_MSC_VER)
  #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
#else
  #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
#endif

#else

#if defined(_MSC_VER)
  #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __FUNCSIG__)
#else
  #define CUTLASS_NOT_IMPLEMENTED() assert(0 && __PRETTY_FUNCTION__)
#endif

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE __forceinline__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
#endif

/// Status code returned by CUTLASS operations
enum class Status {
  kSuccess,                    ///< Operation was successful.
  kErrorMisalignedOperand,     ///< operands fail alignment requirements.
  kErrorInvalidDataType,       ///< DataType fails requirement.
  kErrorInvalidLayout,         ///< Layout fails alignment requirement.
  kErrorInvalidProblem,        ///< Specified problem size is not supported by operator.
  kErrorNotSupported,          ///< Operation is not supported on current device.
  kErrorWorkspaceNull,         ///< The given workspace is null when it is required to be non-null.
  kErrorInternal,              ///< An error within CUTLASS occurred.
  kErrorArchMismatch,          ///< CUTLASS runs on a device that it was not compiled for.
  kErrorInsufficientDriver,    ///< CUTLASS runs with a driver that is too old.
  kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
  kInvalid                     ///< Status is unspecified.
};

/// Convert cutlass status to status strings
CUTLASS_HOST_DEVICE
static char const* cutlassGetStatusString(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return "Success";
    case cutlass::Status::kErrorMisalignedOperand:
      return "Error Misaligned Operand";
    case cutlass::Status::kErrorInvalidDataType:
      return "Error Invalid Data Type";
    case cutlass::Status::kErrorInvalidLayout:
      return "Error Invalid Layout";
    case cutlass::Status::kErrorInvalidProblem:
      return "Error Invalid Problem";
    case cutlass::Status::kErrorNotSupported:
      return "Error Not Supported";
    case cutlass::Status::kErrorWorkspaceNull:
      return "Error Workspace Null";
    case cutlass::Status::kErrorInternal:
      return "Error Internal";
    case cutlass::Status::kErrorInsufficientDriver:
      return "Error Insufficient Driver";
    case cutlass::Status::kErrorArchMismatch:
      return "Error Architecture Mismatch";
    case cutlass::Status::kErrorMemoryAllocation:
      return "Error Memory Allocation failed";
    case cutlass::Status::kInvalid: break;
  }

  return "Invalid status";
}

////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
#define CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 0
#endif


// CUDA 10.1 introduces the mma instruction
#if !defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
#define CUTLASS_ENABLE_TENSOR_CORE_MMA 0
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_ASSERT(x) assert(x)

////////////////////////////////////////////////////////////////////////////////////////////////////

// CUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the CUDA compiler.
#if defined(__CUDA_ARCH__)
  #if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
    #define CUTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define CUTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
  #else
    #define CUTLASS_PRAGMA_UNROLL #pragma unroll
    #define CUTLASS_PRAGMA_NO_UNROLL #pragma unroll 1
  #endif

  #define CUTLASS_GEMM_LOOP CUTLASS_PRAGMA_NO_UNROLL

#else

    #define CUTLASS_PRAGMA_UNROLL
    #define CUTLASS_PRAGMA_NO_UNROLL
    #define CUTLASS_GEMM_LOOP

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

static const int NUM_THREADS_PER_WARP = 32;
static const int NUM_THREADS_PER_HALF_WARP = NUM_THREADS_PER_WARP / 2;
static const int NUM_THREADS_PER_QUAD = 4;
static const int NUM_THREADS_PER_QUAD_PAIR = NUM_THREADS_PER_QUAD * 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function to return true when called by thread 0 of threadblock 0.
CUTLASS_HOST_DEVICE bool thread0() {
  #if defined(__CUDA_ARCH__)
    return (!threadIdx.x && !threadIdx.y && !threadIdx.z) && (!blockIdx.x && !blockIdx.y && !blockIdx.z);
  #else
    return false;
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

