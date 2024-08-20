/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuda.h>

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90.hpp>

#include <cute/container/alignment.hpp>
#include <cute/container/bit_field.hpp>
#include <cute/numeric/half.hpp>  // to_Format<half_t>
#include <cute/numeric/int.hpp>   // to_Format<[u]intX>

namespace cute {

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// Barriers are 64-bit of user-managed information used in broadly two types
/// syncronization patterns 1) arrive/wait on threads (usage: cp.async and
/// warp-specialized kernels) 2) transaction-based (usage: TMA transaction where
/// a CTA issues one transaction)
//////////////////////////////////////////////////////////////////////////////////////////////////////

// Initialize barrier present in shared memory
CUTE_HOST_DEVICE
void initialize_barrier(
    uint64_t& smem_barrier,  // 64 bits user-manged barrier in smem
    int thread_count =
        1)  // Thread count expected to arrive/wait on this barrier
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
               "r"(thread_count));
#endif
}

// Set the number of bytes transfered per transaction
CUTE_HOST_DEVICE
void set_barrier_transaction_bytes(
    uint64_t& smem_barrier,  // 64 bits user-manged barrier in smem
    uint32_t bytes)  // Number of bytes transfered by per TMA transaction
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smem_int_ptr),
      "r"(bytes));
#endif
}

// Barrier wait
CUTE_HOST_DEVICE
void wait_barrier(
    uint64_t& smem_barrier,  // 64 bits user-manged barrier in smem
    int phase_bit)           // Current phase bit the barrier waiting to flip
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));

#endif
}

// Barrier arrive
CUTE_HOST_DEVICE
void arrive_barrier(
    uint64_t& smem_barrier)  // 64 bits user-manged barrier in smem
{
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .b64 state; \n"
      "mbarrier.arrive.shared.b64   state, [%0];\n"
      "}\n" ::"r"(smem_int_ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TMA Descriptor and utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace TMA {

enum class SmemSwizzleBits : uint8_t {
  DISABLE = 0,
  B32 = 1,
  B64 = 2,
  B128 = 3,
};

#if (__CUDACC_VER_MAJOR__ >= 12)

template <class T>
inline CUtensorMapDataType to_CUtensorMapDataType() {
  if constexpr (std::is_same<T, int8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if constexpr (std::is_same<T, uint16_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT16;
  } else if constexpr (std::is_same<T, uint32_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT32;
  } else if constexpr (std::is_same<T, uint64_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_UINT64;
  } else if constexpr (std::is_same<T, int32_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if constexpr (std::is_same<T, int64_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_INT64;
  } else if constexpr (std::is_same<T, half_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr (std::is_same<T, float>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if constexpr (std::is_same<T, double>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
  } else if constexpr (std::is_same<T, bfloat16_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr (std::is_same<T, tfloat32_t>::value) {
    return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
  } else {
    static_assert(sizeof(T) < 0, "Unknown TMA Format!");
  }
}

inline CUtensorMapSwizzle to_CUtensorMapSwizzle(SmemSwizzleBits const& t) {
  switch (t) {
    default:
      assert(false && "Unknown SmemSwizzleBits!");
    case SmemSwizzleBits::DISABLE:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case SmemSwizzleBits::B32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case SmemSwizzleBits::B64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case SmemSwizzleBits::B128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
  }
}

#endif  // (__CUDACC_VER_MAJOR__ >= 12)
}  // end namespace TMA

#if (__CUDACC_VER_MAJOR__ >= 12)
using TmaDescriptor = CUtensorMap;
#else
using TmaDescriptor = struct { char bytes[128]; };
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
/// Initiates a TensorMap Prefetch
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTE_HOST_DEVICE
void prefetch_tma_descriptor(TmaDescriptor const* desc_ptr) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
  // Prefetch TMA Descriptor using generic addressing (i.e. no specific state
  // space: const or param)
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
#else
  CUTE_RUNTIME_ASSERT(
      "Trying to use TMA Descriptor Prefetch without "
      "CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
}

///////////////////////////////////////////////////////////////////////////////

}  // end namespace cute
