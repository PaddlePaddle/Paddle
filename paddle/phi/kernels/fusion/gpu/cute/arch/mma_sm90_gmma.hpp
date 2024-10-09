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

#include <cute/arch/mma.hpp>
#include <cute/config.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && \
     defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define CUTE_ARCH_MMA_SM90_ENABLED
#endif

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Warpgroup sync primitives

CUTE_HOST_DEVICE
void warpgroup_arrive() {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#else
  CUTE_RUNTIME_ASSERT(
      "Attempting to use wgmma.fence without CUTE_ARCH_MMA_SM90_ENABLED");
#endif
}

template <int N>
CUTE_HOST_DEVICE void warpgroup_wait() {
  static_assert(N >= 0 && N <= 7,
                "_warpgroup.wait {N}; must be in range [0, 7]");
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#else
  CUTE_RUNTIME_ASSERT(
      "Attempting to use wgmma.wait_group<N> without "
      "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
}

// Marks the commit point for one or more sized batch of warpgroup MMAs.
CUTE_HOST_DEVICE
void warpgroup_commit_batch() {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#else
  CUTE_RUNTIME_ASSERT(
      "Attempting to use wgmma.commit_group without "
      "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
}

CUTE_HOST_DEVICE
void warpgroup_fence_operand(uint32_t& reg) {
  asm volatile("" : "+r"(reg)::"memory");
}

CUTE_HOST_DEVICE
void warpgroup_fence_operand(float& reg) {
  asm volatile("" : "+f"(reg)::"memory");
}

namespace GMMA {

enum class Major { K = 0, MN = 1 };

enum class ScaleOut { Zero = 0, One = 1 };

enum class ScaleIn { Neg = -1, One = 1 };

}  // namespace GMMA

////////////////////////////////////////////////////////////////////////////////////////////////////
// GMMA PTX definitions:  C = (scaleA * A) * (scaleB * B) + (scaleD * C)
////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
        "{%0, %1},"
        " %2,"
        " %3,"
        " %4, %5, %6, %7, %8;\n"
        : "+r"(d0), "+r"(d1)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[2];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
        "{%0,  %1},"
        "{%2,  %3,  %4,  %5},"
        " %6,"
        " %7,  %8,  %9,  %10;\n"
        : "+r"(d0), "+r"(d1)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6,  %7,  %8,  %9,  %10;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9,  %10, %11, %12;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10, %11, %12, %13, %14;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13, %14, %15, %16;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21, %22, %23, %24;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[24];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23},"
        " %24,"
        " %25,"
        " %26, %27, %28, %29, %30;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[24];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23},"
        "{%24, %25, %26, %27},"
        " %28,"
        " %29, %30, %31, %32;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37, %38, %39, %40;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f16.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50, %51, %52, %53, %54;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f16.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53,  %54,  %55,  %56;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x16_F16F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,  %67,  %68,  %69,  %70;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x16_F16F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x16 F16+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x16_F16F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f16.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69,  %70,  %71,  %72;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x16_F16F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6,  %7,  %8,  %9,  %10;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9,  %10, %11, %12;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3,
                                   float& d4,
                                   float& d5,
                                   float& d6,
                                   float& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10, %11, %12, %13, %14;\n"
        : "+f"(d0),
          "+f"(d1),
          "+f"(d2),
          "+f"(d3),
          "+f"(d4),
          "+f"(d5),
          "+f"(d6),
          "+f"(d7)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[8];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3,
                                   float& d4,
                                   float& d5,
                                   float& d6,
                                   float& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13, %14, %15, %16;\n"
        : "+f"(d0),
          "+f"(d1),
          "+f"(d2),
          "+f"(d3),
          "+f"(d4),
          "+f"(d5),
          "+f"(d6),
          "+f"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[16];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21, %22, %23, %24;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[32];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37, %38, %39, %40;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50, %51, %52, %53, %54;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[48];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53,  %54,  %55,  %56;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,  %67,  %68,  %69,  %70;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69,  %70,  %71,  %72;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63,
                                   float& d64,
                                   float& d65,
                                   float& d66,
                                   float& d67,
                                   float& d68,
                                   float& d69,
                                   float& d70,
                                   float& d71,
                                   float& d72,
                                   float& d73,
                                   float& d74,
                                   float& d75,
                                   float& d76,
                                   float& d77,
                                   float& d78,
                                   float& d79,
                                   float& d80,
                                   float& d81,
                                   float& d82,
                                   float& d83,
                                   float& d84,
                                   float& d85,
                                   float& d86,
                                   float& d87,
                                   float& d88,
                                   float& d89,
                                   float& d90,
                                   float& d91,
                                   float& d92,
                                   float& d93,
                                   float& d94,
                                   float& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98,  %99,  %100, %101, %102;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63),
          "+f"(d64),
          "+f"(d65),
          "+f"(d66),
          "+f"(d67),
          "+f"(d68),
          "+f"(d69),
          "+f"(d70),
          "+f"(d71),
          "+f"(d72),
          "+f"(d73),
          "+f"(d74),
          "+f"(d75),
          "+f"(d76),
          "+f"(d77),
          "+f"(d78),
          "+f"(d79),
          "+f"(d80),
          "+f"(d81),
          "+f"(d82),
          "+f"(d83),
          "+f"(d84),
          "+f"(d85),
          "+f"(d86),
          "+f"(d87),
          "+f"(d88),
          "+f"(d89),
          "+f"(d90),
          "+f"(d91),
          "+f"(d92),
          "+f"(d93),
          "+f"(d94),
          "+f"(d95)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[96];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63,
                                   float& d64,
                                   float& d65,
                                   float& d66,
                                   float& d67,
                                   float& d68,
                                   float& d69,
                                   float& d70,
                                   float& d71,
                                   float& d72,
                                   float& d73,
                                   float& d74,
                                   float& d75,
                                   float& d76,
                                   float& d77,
                                   float& d78,
                                   float& d79,
                                   float& d80,
                                   float& d81,
                                   float& d82,
                                   float& d83,
                                   float& d84,
                                   float& d85,
                                   float& d86,
                                   float& d87,
                                   float& d88,
                                   float& d89,
                                   float& d90,
                                   float& d91,
                                   float& d92,
                                   float& d93,
                                   float& d94,
                                   float& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101, %102, %103, %104;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63),
          "+f"(d64),
          "+f"(d65),
          "+f"(d66),
          "+f"(d67),
          "+f"(d68),
          "+f"(d69),
          "+f"(d70),
          "+f"(d71),
          "+f"(d72),
          "+f"(d73),
          "+f"(d74),
          "+f"(d75),
          "+f"(d76),
          "+f"(d77),
          "+f"(d78),
          "+f"(d79),
          "+f"(d80),
          "+f"(d81),
          "+f"(d82),
          "+f"(d83),
          "+f"(d84),
          "+f"(d85),
          "+f"(d86),
          "+f"(d87),
          "+f"(d88),
          "+f"(d89),
          "+f"(d90),
          "+f"(d91),
          "+f"(d92),
          "+f"(d93),
          "+f"(d94),
          "+f"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x16_F32F16F16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d000,
                                   float& d001,
                                   float& d002,
                                   float& d003,
                                   float& d004,
                                   float& d005,
                                   float& d006,
                                   float& d007,
                                   float& d008,
                                   float& d009,
                                   float& d010,
                                   float& d011,
                                   float& d012,
                                   float& d013,
                                   float& d014,
                                   float& d015,
                                   float& d016,
                                   float& d017,
                                   float& d018,
                                   float& d019,
                                   float& d020,
                                   float& d021,
                                   float& d022,
                                   float& d023,
                                   float& d024,
                                   float& d025,
                                   float& d026,
                                   float& d027,
                                   float& d028,
                                   float& d029,
                                   float& d030,
                                   float& d031,
                                   float& d032,
                                   float& d033,
                                   float& d034,
                                   float& d035,
                                   float& d036,
                                   float& d037,
                                   float& d038,
                                   float& d039,
                                   float& d040,
                                   float& d041,
                                   float& d042,
                                   float& d043,
                                   float& d044,
                                   float& d045,
                                   float& d046,
                                   float& d047,
                                   float& d048,
                                   float& d049,
                                   float& d050,
                                   float& d051,
                                   float& d052,
                                   float& d053,
                                   float& d054,
                                   float& d055,
                                   float& d056,
                                   float& d057,
                                   float& d058,
                                   float& d059,
                                   float& d060,
                                   float& d061,
                                   float& d062,
                                   float& d063,
                                   float& d064,
                                   float& d065,
                                   float& d066,
                                   float& d067,
                                   float& d068,
                                   float& d069,
                                   float& d070,
                                   float& d071,
                                   float& d072,
                                   float& d073,
                                   float& d074,
                                   float& d075,
                                   float& d076,
                                   float& d077,
                                   float& d078,
                                   float& d079,
                                   float& d080,
                                   float& d081,
                                   float& d082,
                                   float& d083,
                                   float& d084,
                                   float& d085,
                                   float& d086,
                                   float& d087,
                                   float& d088,
                                   float& d089,
                                   float& d090,
                                   float& d091,
                                   float& d092,
                                   float& d093,
                                   float& d094,
                                   float& d095,
                                   float& d096,
                                   float& d097,
                                   float& d098,
                                   float& d099,
                                   float& d100,
                                   float& d101,
                                   float& d102,
                                   float& d103,
                                   float& d104,
                                   float& d105,
                                   float& d106,
                                   float& d107,
                                   float& d108,
                                   float& d109,
                                   float& d110,
                                   float& d111,
                                   float& d112,
                                   float& d113,
                                   float& d114,
                                   float& d115,
                                   float& d116,
                                   float& d117,
                                   float& d118,
                                   float& d119,
                                   float& d120,
                                   float& d121,
                                   float& d122,
                                   float& d123,
                                   float& d124,
                                   float& d125,
                                   float& d126,
                                   float& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130, %131, %132, %133, %134;\n"
        : "+f"(d000),
          "+f"(d001),
          "+f"(d002),
          "+f"(d003),
          "+f"(d004),
          "+f"(d005),
          "+f"(d006),
          "+f"(d007),
          "+f"(d008),
          "+f"(d009),
          "+f"(d010),
          "+f"(d011),
          "+f"(d012),
          "+f"(d013),
          "+f"(d014),
          "+f"(d015),
          "+f"(d016),
          "+f"(d017),
          "+f"(d018),
          "+f"(d019),
          "+f"(d020),
          "+f"(d021),
          "+f"(d022),
          "+f"(d023),
          "+f"(d024),
          "+f"(d025),
          "+f"(d026),
          "+f"(d027),
          "+f"(d028),
          "+f"(d029),
          "+f"(d030),
          "+f"(d031),
          "+f"(d032),
          "+f"(d033),
          "+f"(d034),
          "+f"(d035),
          "+f"(d036),
          "+f"(d037),
          "+f"(d038),
          "+f"(d039),
          "+f"(d040),
          "+f"(d041),
          "+f"(d042),
          "+f"(d043),
          "+f"(d044),
          "+f"(d045),
          "+f"(d046),
          "+f"(d047),
          "+f"(d048),
          "+f"(d049),
          "+f"(d050),
          "+f"(d051),
          "+f"(d052),
          "+f"(d053),
          "+f"(d054),
          "+f"(d055),
          "+f"(d056),
          "+f"(d057),
          "+f"(d058),
          "+f"(d059),
          "+f"(d060),
          "+f"(d061),
          "+f"(d062),
          "+f"(d063),
          "+f"(d064),
          "+f"(d065),
          "+f"(d066),
          "+f"(d067),
          "+f"(d068),
          "+f"(d069),
          "+f"(d070),
          "+f"(d071),
          "+f"(d072),
          "+f"(d073),
          "+f"(d074),
          "+f"(d075),
          "+f"(d076),
          "+f"(d077),
          "+f"(d078),
          "+f"(d079),
          "+f"(d080),
          "+f"(d081),
          "+f"(d082),
          "+f"(d083),
          "+f"(d084),
          "+f"(d085),
          "+f"(d086),
          "+f"(d087),
          "+f"(d088),
          "+f"(d089),
          "+f"(d090),
          "+f"(d091),
          "+f"(d092),
          "+f"(d093),
          "+f"(d094),
          "+f"(d095),
          "+f"(d096),
          "+f"(d097),
          "+f"(d098),
          "+f"(d099),
          "+f"(d100),
          "+f"(d101),
          "+f"(d102),
          "+f"(d103),
          "+f"(d104),
          "+f"(d105),
          "+f"(d106),
          "+f"(d107),
          "+f"(d108),
          "+f"(d109),
          "+f"(d110),
          "+f"(d111),
          "+f"(d112),
          "+f"(d113),
          "+f"(d114),
          "+f"(d115),
          "+f"(d116),
          "+f"(d117),
          "+f"(d118),
          "+f"(d119),
          "+f"(d120),
          "+f"(d121),
          "+f"(d122),
          "+f"(d123),
          "+f"(d124),
          "+f"(d125),
          "+f"(d126),
          "+f"(d127)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x16_F32F16F16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x16 F32+=F16*F16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x16_F32F16F16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[128];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   float& d000,
                                   float& d001,
                                   float& d002,
                                   float& d003,
                                   float& d004,
                                   float& d005,
                                   float& d006,
                                   float& d007,
                                   float& d008,
                                   float& d009,
                                   float& d010,
                                   float& d011,
                                   float& d012,
                                   float& d013,
                                   float& d014,
                                   float& d015,
                                   float& d016,
                                   float& d017,
                                   float& d018,
                                   float& d019,
                                   float& d020,
                                   float& d021,
                                   float& d022,
                                   float& d023,
                                   float& d024,
                                   float& d025,
                                   float& d026,
                                   float& d027,
                                   float& d028,
                                   float& d029,
                                   float& d030,
                                   float& d031,
                                   float& d032,
                                   float& d033,
                                   float& d034,
                                   float& d035,
                                   float& d036,
                                   float& d037,
                                   float& d038,
                                   float& d039,
                                   float& d040,
                                   float& d041,
                                   float& d042,
                                   float& d043,
                                   float& d044,
                                   float& d045,
                                   float& d046,
                                   float& d047,
                                   float& d048,
                                   float& d049,
                                   float& d050,
                                   float& d051,
                                   float& d052,
                                   float& d053,
                                   float& d054,
                                   float& d055,
                                   float& d056,
                                   float& d057,
                                   float& d058,
                                   float& d059,
                                   float& d060,
                                   float& d061,
                                   float& d062,
                                   float& d063,
                                   float& d064,
                                   float& d065,
                                   float& d066,
                                   float& d067,
                                   float& d068,
                                   float& d069,
                                   float& d070,
                                   float& d071,
                                   float& d072,
                                   float& d073,
                                   float& d074,
                                   float& d075,
                                   float& d076,
                                   float& d077,
                                   float& d078,
                                   float& d079,
                                   float& d080,
                                   float& d081,
                                   float& d082,
                                   float& d083,
                                   float& d084,
                                   float& d085,
                                   float& d086,
                                   float& d087,
                                   float& d088,
                                   float& d089,
                                   float& d090,
                                   float& d091,
                                   float& d092,
                                   float& d093,
                                   float& d094,
                                   float& d095,
                                   float& d096,
                                   float& d097,
                                   float& d098,
                                   float& d099,
                                   float& d100,
                                   float& d101,
                                   float& d102,
                                   float& d103,
                                   float& d104,
                                   float& d105,
                                   float& d106,
                                   float& d107,
                                   float& d108,
                                   float& d109,
                                   float& d110,
                                   float& d111,
                                   float& d112,
                                   float& d113,
                                   float& d114,
                                   float& d115,
                                   float& d116,
                                   float& d117,
                                   float& d118,
                                   float& d119,
                                   float& d120,
                                   float& d121,
                                   float& d122,
                                   float& d123,
                                   float& d124,
                                   float& d125,
                                   float& d126,
                                   float& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133, %134, %135, %136;\n"
        : "+f"(d000),
          "+f"(d001),
          "+f"(d002),
          "+f"(d003),
          "+f"(d004),
          "+f"(d005),
          "+f"(d006),
          "+f"(d007),
          "+f"(d008),
          "+f"(d009),
          "+f"(d010),
          "+f"(d011),
          "+f"(d012),
          "+f"(d013),
          "+f"(d014),
          "+f"(d015),
          "+f"(d016),
          "+f"(d017),
          "+f"(d018),
          "+f"(d019),
          "+f"(d020),
          "+f"(d021),
          "+f"(d022),
          "+f"(d023),
          "+f"(d024),
          "+f"(d025),
          "+f"(d026),
          "+f"(d027),
          "+f"(d028),
          "+f"(d029),
          "+f"(d030),
          "+f"(d031),
          "+f"(d032),
          "+f"(d033),
          "+f"(d034),
          "+f"(d035),
          "+f"(d036),
          "+f"(d037),
          "+f"(d038),
          "+f"(d039),
          "+f"(d040),
          "+f"(d041),
          "+f"(d042),
          "+f"(d043),
          "+f"(d044),
          "+f"(d045),
          "+f"(d046),
          "+f"(d047),
          "+f"(d048),
          "+f"(d049),
          "+f"(d050),
          "+f"(d051),
          "+f"(d052),
          "+f"(d053),
          "+f"(d054),
          "+f"(d055),
          "+f"(d056),
          "+f"(d057),
          "+f"(d058),
          "+f"(d059),
          "+f"(d060),
          "+f"(d061),
          "+f"(d062),
          "+f"(d063),
          "+f"(d064),
          "+f"(d065),
          "+f"(d066),
          "+f"(d067),
          "+f"(d068),
          "+f"(d069),
          "+f"(d070),
          "+f"(d071),
          "+f"(d072),
          "+f"(d073),
          "+f"(d074),
          "+f"(d075),
          "+f"(d076),
          "+f"(d077),
          "+f"(d078),
          "+f"(d079),
          "+f"(d080),
          "+f"(d081),
          "+f"(d082),
          "+f"(d083),
          "+f"(d084),
          "+f"(d085),
          "+f"(d086),
          "+f"(d087),
          "+f"(d088),
          "+f"(d089),
          "+f"(d090),
          "+f"(d091),
          "+f"(d092),
          "+f"(d093),
          "+f"(d094),
          "+f"(d095),
          "+f"(d096),
          "+f"(d097),
          "+f"(d098),
          "+f"(d099),
          "+f"(d100),
          "+f"(d101),
          "+f"(d102),
          "+f"(d103),
          "+f"(d104),
          "+f"(d105),
          "+f"(d106),
          "+f"(d107),
          "+f"(d108),
          "+f"(d109),
          "+f"(d110),
          "+f"(d111),
          "+f"(d112),
          "+f"(d113),
          "+f"(d114),
          "+f"(d115),
          "+f"(d116),
          "+f"(d117),
          "+f"(d118),
          "+f"(d119),
          "+f"(d120),
          "+f"(d121),
          "+f"(d122),
          "+f"(d123),
          "+f"(d124),
          "+f"(d125),
          "+f"(d126),
          "+f"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x16_F32F16F16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6,  %7,  %8,  %9,  %10;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9,  %10, %11, %12;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3,
                                   float& d4,
                                   float& d5,
                                   float& d6,
                                   float& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10, %11, %12, %13, %14;\n"
        : "+f"(d0),
          "+f"(d1),
          "+f"(d2),
          "+f"(d3),
          "+f"(d4),
          "+f"(d5),
          "+f"(d6),
          "+f"(d7)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[8];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3,
                                   float& d4,
                                   float& d5,
                                   float& d6,
                                   float& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13, %14, %15, %16;\n"
        : "+f"(d0),
          "+f"(d1),
          "+f"(d2),
          "+f"(d3),
          "+f"(d4),
          "+f"(d5),
          "+f"(d6),
          "+f"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18, %19, %20, %21, %22;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[16];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21, %22, %23, %24;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[32];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37, %38, %39, %40;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50, %51, %52, %53, %54;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[48];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53,  %54,  %55,  %56;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,  %67,  %68,  %69,  %70;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69,  %70,  %71,  %72;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63,
                                   float& d64,
                                   float& d65,
                                   float& d66,
                                   float& d67,
                                   float& d68,
                                   float& d69,
                                   float& d70,
                                   float& d71,
                                   float& d72,
                                   float& d73,
                                   float& d74,
                                   float& d75,
                                   float& d76,
                                   float& d77,
                                   float& d78,
                                   float& d79,
                                   float& d80,
                                   float& d81,
                                   float& d82,
                                   float& d83,
                                   float& d84,
                                   float& d85,
                                   float& d86,
                                   float& d87,
                                   float& d88,
                                   float& d89,
                                   float& d90,
                                   float& d91,
                                   float& d92,
                                   float& d93,
                                   float& d94,
                                   float& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98,  %99,  %100, %101, %102;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63),
          "+f"(d64),
          "+f"(d65),
          "+f"(d66),
          "+f"(d67),
          "+f"(d68),
          "+f"(d69),
          "+f"(d70),
          "+f"(d71),
          "+f"(d72),
          "+f"(d73),
          "+f"(d74),
          "+f"(d75),
          "+f"(d76),
          "+f"(d77),
          "+f"(d78),
          "+f"(d79),
          "+f"(d80),
          "+f"(d81),
          "+f"(d82),
          "+f"(d83),
          "+f"(d84),
          "+f"(d85),
          "+f"(d86),
          "+f"(d87),
          "+f"(d88),
          "+f"(d89),
          "+f"(d90),
          "+f"(d91),
          "+f"(d92),
          "+f"(d93),
          "+f"(d94),
          "+f"(d95)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[96];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63,
                                   float& d64,
                                   float& d65,
                                   float& d66,
                                   float& d67,
                                   float& d68,
                                   float& d69,
                                   float& d70,
                                   float& d71,
                                   float& d72,
                                   float& d73,
                                   float& d74,
                                   float& d75,
                                   float& d76,
                                   float& d77,
                                   float& d78,
                                   float& d79,
                                   float& d80,
                                   float& d81,
                                   float& d82,
                                   float& d83,
                                   float& d84,
                                   float& d85,
                                   float& d86,
                                   float& d87,
                                   float& d88,
                                   float& d89,
                                   float& d90,
                                   float& d91,
                                   float& d92,
                                   float& d93,
                                   float& d94,
                                   float& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101, %102, %103, %104;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63),
          "+f"(d64),
          "+f"(d65),
          "+f"(d66),
          "+f"(d67),
          "+f"(d68),
          "+f"(d69),
          "+f"(d70),
          "+f"(d71),
          "+f"(d72),
          "+f"(d73),
          "+f"(d74),
          "+f"(d75),
          "+f"(d76),
          "+f"(d77),
          "+f"(d78),
          "+f"(d79),
          "+f"(d80),
          "+f"(d81),
          "+f"(d82),
          "+f"(d83),
          "+f"(d84),
          "+f"(d85),
          "+f"(d86),
          "+f"(d87),
          "+f"(d88),
          "+f"(d89),
          "+f"(d90),
          "+f"(d91),
          "+f"(d92),
          "+f"(d93),
          "+f"(d94),
          "+f"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x16_F32BF16BF16_SS {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d000,
                                   float& d001,
                                   float& d002,
                                   float& d003,
                                   float& d004,
                                   float& d005,
                                   float& d006,
                                   float& d007,
                                   float& d008,
                                   float& d009,
                                   float& d010,
                                   float& d011,
                                   float& d012,
                                   float& d013,
                                   float& d014,
                                   float& d015,
                                   float& d016,
                                   float& d017,
                                   float& d018,
                                   float& d019,
                                   float& d020,
                                   float& d021,
                                   float& d022,
                                   float& d023,
                                   float& d024,
                                   float& d025,
                                   float& d026,
                                   float& d027,
                                   float& d028,
                                   float& d029,
                                   float& d030,
                                   float& d031,
                                   float& d032,
                                   float& d033,
                                   float& d034,
                                   float& d035,
                                   float& d036,
                                   float& d037,
                                   float& d038,
                                   float& d039,
                                   float& d040,
                                   float& d041,
                                   float& d042,
                                   float& d043,
                                   float& d044,
                                   float& d045,
                                   float& d046,
                                   float& d047,
                                   float& d048,
                                   float& d049,
                                   float& d050,
                                   float& d051,
                                   float& d052,
                                   float& d053,
                                   float& d054,
                                   float& d055,
                                   float& d056,
                                   float& d057,
                                   float& d058,
                                   float& d059,
                                   float& d060,
                                   float& d061,
                                   float& d062,
                                   float& d063,
                                   float& d064,
                                   float& d065,
                                   float& d066,
                                   float& d067,
                                   float& d068,
                                   float& d069,
                                   float& d070,
                                   float& d071,
                                   float& d072,
                                   float& d073,
                                   float& d074,
                                   float& d075,
                                   float& d076,
                                   float& d077,
                                   float& d078,
                                   float& d079,
                                   float& d080,
                                   float& d081,
                                   float& d082,
                                   float& d083,
                                   float& d084,
                                   float& d085,
                                   float& d086,
                                   float& d087,
                                   float& d088,
                                   float& d089,
                                   float& d090,
                                   float& d091,
                                   float& d092,
                                   float& d093,
                                   float& d094,
                                   float& d095,
                                   float& d096,
                                   float& d097,
                                   float& d098,
                                   float& d099,
                                   float& d100,
                                   float& d101,
                                   float& d102,
                                   float& d103,
                                   float& d104,
                                   float& d105,
                                   float& d106,
                                   float& d107,
                                   float& d108,
                                   float& d109,
                                   float& d110,
                                   float& d111,
                                   float& d112,
                                   float& d113,
                                   float& d114,
                                   float& d115,
                                   float& d116,
                                   float& d117,
                                   float& d118,
                                   float& d119,
                                   float& d120,
                                   float& d121,
                                   float& d122,
                                   float& d123,
                                   float& d124,
                                   float& d125,
                                   float& d126,
                                   float& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130, %131, %132, %133, %134;\n"
        : "+f"(d000),
          "+f"(d001),
          "+f"(d002),
          "+f"(d003),
          "+f"(d004),
          "+f"(d005),
          "+f"(d006),
          "+f"(d007),
          "+f"(d008),
          "+f"(d009),
          "+f"(d010),
          "+f"(d011),
          "+f"(d012),
          "+f"(d013),
          "+f"(d014),
          "+f"(d015),
          "+f"(d016),
          "+f"(d017),
          "+f"(d018),
          "+f"(d019),
          "+f"(d020),
          "+f"(d021),
          "+f"(d022),
          "+f"(d023),
          "+f"(d024),
          "+f"(d025),
          "+f"(d026),
          "+f"(d027),
          "+f"(d028),
          "+f"(d029),
          "+f"(d030),
          "+f"(d031),
          "+f"(d032),
          "+f"(d033),
          "+f"(d034),
          "+f"(d035),
          "+f"(d036),
          "+f"(d037),
          "+f"(d038),
          "+f"(d039),
          "+f"(d040),
          "+f"(d041),
          "+f"(d042),
          "+f"(d043),
          "+f"(d044),
          "+f"(d045),
          "+f"(d046),
          "+f"(d047),
          "+f"(d048),
          "+f"(d049),
          "+f"(d050),
          "+f"(d051),
          "+f"(d052),
          "+f"(d053),
          "+f"(d054),
          "+f"(d055),
          "+f"(d056),
          "+f"(d057),
          "+f"(d058),
          "+f"(d059),
          "+f"(d060),
          "+f"(d061),
          "+f"(d062),
          "+f"(d063),
          "+f"(d064),
          "+f"(d065),
          "+f"(d066),
          "+f"(d067),
          "+f"(d068),
          "+f"(d069),
          "+f"(d070),
          "+f"(d071),
          "+f"(d072),
          "+f"(d073),
          "+f"(d074),
          "+f"(d075),
          "+f"(d076),
          "+f"(d077),
          "+f"(d078),
          "+f"(d079),
          "+f"(d080),
          "+f"(d081),
          "+f"(d082),
          "+f"(d083),
          "+f"(d084),
          "+f"(d085),
          "+f"(d086),
          "+f"(d087),
          "+f"(d088),
          "+f"(d089),
          "+f"(d090),
          "+f"(d091),
          "+f"(d092),
          "+f"(d093),
          "+f"(d094),
          "+f"(d095),
          "+f"(d096),
          "+f"(d097),
          "+f"(d098),
          "+f"(d099),
          "+f"(d100),
          "+f"(d101),
          "+f"(d102),
          "+f"(d103),
          "+f"(d104),
          "+f"(d105),
          "+f"(d106),
          "+f"(d107),
          "+f"(d108),
          "+f"(d109),
          "+f"(d110),
          "+f"(d111),
          "+f"(d112),
          "+f"(d113),
          "+f"(d114),
          "+f"(d115),
          "+f"(d116),
          "+f"(d117),
          "+f"(d118),
          "+f"(d119),
          "+f"(d120),
          "+f"(d121),
          "+f"(d122),
          "+f"(d123),
          "+f"(d124),
          "+f"(d125),
          "+f"(d126),
          "+f"(d127)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspA)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x16_F32BF16BF16_SS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x16 F32+=BF16*BF16
template <GMMA::Major tnspA,
          GMMA::Major tnspB,
          GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x16_F32BF16BF16_RS {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[128];

  static_assert(tnspA == GMMA::Major::K,
                "Register source operand A must have K major layout.");

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   float& d000,
                                   float& d001,
                                   float& d002,
                                   float& d003,
                                   float& d004,
                                   float& d005,
                                   float& d006,
                                   float& d007,
                                   float& d008,
                                   float& d009,
                                   float& d010,
                                   float& d011,
                                   float& d012,
                                   float& d013,
                                   float& d014,
                                   float& d015,
                                   float& d016,
                                   float& d017,
                                   float& d018,
                                   float& d019,
                                   float& d020,
                                   float& d021,
                                   float& d022,
                                   float& d023,
                                   float& d024,
                                   float& d025,
                                   float& d026,
                                   float& d027,
                                   float& d028,
                                   float& d029,
                                   float& d030,
                                   float& d031,
                                   float& d032,
                                   float& d033,
                                   float& d034,
                                   float& d035,
                                   float& d036,
                                   float& d037,
                                   float& d038,
                                   float& d039,
                                   float& d040,
                                   float& d041,
                                   float& d042,
                                   float& d043,
                                   float& d044,
                                   float& d045,
                                   float& d046,
                                   float& d047,
                                   float& d048,
                                   float& d049,
                                   float& d050,
                                   float& d051,
                                   float& d052,
                                   float& d053,
                                   float& d054,
                                   float& d055,
                                   float& d056,
                                   float& d057,
                                   float& d058,
                                   float& d059,
                                   float& d060,
                                   float& d061,
                                   float& d062,
                                   float& d063,
                                   float& d064,
                                   float& d065,
                                   float& d066,
                                   float& d067,
                                   float& d068,
                                   float& d069,
                                   float& d070,
                                   float& d071,
                                   float& d072,
                                   float& d073,
                                   float& d074,
                                   float& d075,
                                   float& d076,
                                   float& d077,
                                   float& d078,
                                   float& d079,
                                   float& d080,
                                   float& d081,
                                   float& d082,
                                   float& d083,
                                   float& d084,
                                   float& d085,
                                   float& d086,
                                   float& d087,
                                   float& d088,
                                   float& d089,
                                   float& d090,
                                   float& d091,
                                   float& d092,
                                   float& d093,
                                   float& d094,
                                   float& d095,
                                   float& d096,
                                   float& d097,
                                   float& d098,
                                   float& d099,
                                   float& d100,
                                   float& d101,
                                   float& d102,
                                   float& d103,
                                   float& d104,
                                   float& d105,
                                   float& d106,
                                   float& d107,
                                   float& d108,
                                   float& d109,
                                   float& d110,
                                   float& d111,
                                   float& d112,
                                   float& d113,
                                   float& d114,
                                   float& d115,
                                   float& d116,
                                   float& d117,
                                   float& d118,
                                   float& d119,
                                   float& d120,
                                   float& d121,
                                   float& d122,
                                   float& d123,
                                   float& d124,
                                   float& d125,
                                   float& d126,
                                   float& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133, %134, %135, %136;\n"
        : "+f"(d000),
          "+f"(d001),
          "+f"(d002),
          "+f"(d003),
          "+f"(d004),
          "+f"(d005),
          "+f"(d006),
          "+f"(d007),
          "+f"(d008),
          "+f"(d009),
          "+f"(d010),
          "+f"(d011),
          "+f"(d012),
          "+f"(d013),
          "+f"(d014),
          "+f"(d015),
          "+f"(d016),
          "+f"(d017),
          "+f"(d018),
          "+f"(d019),
          "+f"(d020),
          "+f"(d021),
          "+f"(d022),
          "+f"(d023),
          "+f"(d024),
          "+f"(d025),
          "+f"(d026),
          "+f"(d027),
          "+f"(d028),
          "+f"(d029),
          "+f"(d030),
          "+f"(d031),
          "+f"(d032),
          "+f"(d033),
          "+f"(d034),
          "+f"(d035),
          "+f"(d036),
          "+f"(d037),
          "+f"(d038),
          "+f"(d039),
          "+f"(d040),
          "+f"(d041),
          "+f"(d042),
          "+f"(d043),
          "+f"(d044),
          "+f"(d045),
          "+f"(d046),
          "+f"(d047),
          "+f"(d048),
          "+f"(d049),
          "+f"(d050),
          "+f"(d051),
          "+f"(d052),
          "+f"(d053),
          "+f"(d054),
          "+f"(d055),
          "+f"(d056),
          "+f"(d057),
          "+f"(d058),
          "+f"(d059),
          "+f"(d060),
          "+f"(d061),
          "+f"(d062),
          "+f"(d063),
          "+f"(d064),
          "+f"(d065),
          "+f"(d066),
          "+f"(d067),
          "+f"(d068),
          "+f"(d069),
          "+f"(d070),
          "+f"(d071),
          "+f"(d072),
          "+f"(d073),
          "+f"(d074),
          "+f"(d075),
          "+f"(d076),
          "+f"(d077),
          "+f"(d078),
          "+f"(d079),
          "+f"(d080),
          "+f"(d081),
          "+f"(d082),
          "+f"(d083),
          "+f"(d084),
          "+f"(d085),
          "+f"(d086),
          "+f"(d087),
          "+f"(d088),
          "+f"(d089),
          "+f"(d090),
          "+f"(d091),
          "+f"(d092),
          "+f"(d093),
          "+f"(d094),
          "+f"(d095),
          "+f"(d096),
          "+f"(d097),
          "+f"(d098),
          "+f"(d099),
          "+f"(d100),
          "+f"(d101),
          "+f"(d102),
          "+f"(d103),
          "+f"(d104),
          "+f"(d105),
          "+f"(d106),
          "+f"(d107),
          "+f"(d108),
          "+f"(d109),
          "+f"(d110),
          "+f"(d111),
          "+f"(d112),
          "+f"(d113),
          "+f"(d114),
          "+f"(d115),
          "+f"(d116),
          "+f"(d117),
          "+f"(d118),
          "+f"(d119),
          "+f"(d120),
          "+f"(d121),
          "+f"(d122),
          "+f"(d123),
          "+f"(d124),
          "+f"(d125),
          "+f"(d126),
          "+f"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)),
          "n"(int32_t(tnspB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x16_F32BF16BF16_RS without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6,  %7,  %8;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x8x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x8x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9,  %10, %11;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3,
                                   float& d4,
                                   float& d5,
                                   float& d6,
                                   float& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10, %11, %12;\n"
        : "+f"(d0),
          "+f"(d1),
          "+f"(d2),
          "+f"(d3),
          "+f"(d4),
          "+f"(d5),
          "+f"(d6),
          "+f"(d7)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x16x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x16x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   float& d0,
                                   float& d1,
                                   float& d2,
                                   float& d3,
                                   float& d4,
                                   float& d5,
                                   float& d6,
                                   float& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13, %14, %15;\n"
        : "+f"(d0),
          "+f"(d1),
          "+f"(d2),
          "+f"(d3),
          "+f"(d4),
          "+f"(d5),
          "+f"(d6),
          "+f"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18, %19, %20;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x32x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x32x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21, %22, %23;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34, %35, %36;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x64x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37, %38, %39;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50, %51, %52;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x96x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x96x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53,  %54,  %55;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66,  %67,  %68;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x128x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x128x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69,  %70,  %71;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63,
                                   float& d64,
                                   float& d65,
                                   float& d66,
                                   float& d67,
                                   float& d68,
                                   float& d69,
                                   float& d70,
                                   float& d71,
                                   float& d72,
                                   float& d73,
                                   float& d74,
                                   float& d75,
                                   float& d76,
                                   float& d77,
                                   float& d78,
                                   float& d79,
                                   float& d80,
                                   float& d81,
                                   float& d82,
                                   float& d83,
                                   float& d84,
                                   float& d85,
                                   float& d86,
                                   float& d87,
                                   float& d88,
                                   float& d89,
                                   float& d90,
                                   float& d91,
                                   float& d92,
                                   float& d93,
                                   float& d94,
                                   float& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98,  %99,  %100;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63),
          "+f"(d64),
          "+f"(d65),
          "+f"(d66),
          "+f"(d67),
          "+f"(d68),
          "+f"(d69),
          "+f"(d70),
          "+f"(d71),
          "+f"(d72),
          "+f"(d73),
          "+f"(d74),
          "+f"(d75),
          "+f"(d76),
          "+f"(d77),
          "+f"(d78),
          "+f"(d79),
          "+f"(d80),
          "+f"(d81),
          "+f"(d82),
          "+f"(d83),
          "+f"(d84),
          "+f"(d85),
          "+f"(d86),
          "+f"(d87),
          "+f"(d88),
          "+f"(d89),
          "+f"(d90),
          "+f"(d91),
          "+f"(d92),
          "+f"(d93),
          "+f"(d94),
          "+f"(d95)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x192x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x192x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   float& d00,
                                   float& d01,
                                   float& d02,
                                   float& d03,
                                   float& d04,
                                   float& d05,
                                   float& d06,
                                   float& d07,
                                   float& d08,
                                   float& d09,
                                   float& d10,
                                   float& d11,
                                   float& d12,
                                   float& d13,
                                   float& d14,
                                   float& d15,
                                   float& d16,
                                   float& d17,
                                   float& d18,
                                   float& d19,
                                   float& d20,
                                   float& d21,
                                   float& d22,
                                   float& d23,
                                   float& d24,
                                   float& d25,
                                   float& d26,
                                   float& d27,
                                   float& d28,
                                   float& d29,
                                   float& d30,
                                   float& d31,
                                   float& d32,
                                   float& d33,
                                   float& d34,
                                   float& d35,
                                   float& d36,
                                   float& d37,
                                   float& d38,
                                   float& d39,
                                   float& d40,
                                   float& d41,
                                   float& d42,
                                   float& d43,
                                   float& d44,
                                   float& d45,
                                   float& d46,
                                   float& d47,
                                   float& d48,
                                   float& d49,
                                   float& d50,
                                   float& d51,
                                   float& d52,
                                   float& d53,
                                   float& d54,
                                   float& d55,
                                   float& d56,
                                   float& d57,
                                   float& d58,
                                   float& d59,
                                   float& d60,
                                   float& d61,
                                   float& d62,
                                   float& d63,
                                   float& d64,
                                   float& d65,
                                   float& d66,
                                   float& d67,
                                   float& d68,
                                   float& d69,
                                   float& d70,
                                   float& d71,
                                   float& d72,
                                   float& d73,
                                   float& d74,
                                   float& d75,
                                   float& d76,
                                   float& d77,
                                   float& d78,
                                   float& d79,
                                   float& d80,
                                   float& d81,
                                   float& d82,
                                   float& d83,
                                   float& d84,
                                   float& d85,
                                   float& d86,
                                   float& d87,
                                   float& d88,
                                   float& d89,
                                   float& d90,
                                   float& d91,
                                   float& d92,
                                   float& d93,
                                   float& d94,
                                   float& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101, %102, %103;\n"
        : "+f"(d00),
          "+f"(d01),
          "+f"(d02),
          "+f"(d03),
          "+f"(d04),
          "+f"(d05),
          "+f"(d06),
          "+f"(d07),
          "+f"(d08),
          "+f"(d09),
          "+f"(d10),
          "+f"(d11),
          "+f"(d12),
          "+f"(d13),
          "+f"(d14),
          "+f"(d15),
          "+f"(d16),
          "+f"(d17),
          "+f"(d18),
          "+f"(d19),
          "+f"(d20),
          "+f"(d21),
          "+f"(d22),
          "+f"(d23),
          "+f"(d24),
          "+f"(d25),
          "+f"(d26),
          "+f"(d27),
          "+f"(d28),
          "+f"(d29),
          "+f"(d30),
          "+f"(d31),
          "+f"(d32),
          "+f"(d33),
          "+f"(d34),
          "+f"(d35),
          "+f"(d36),
          "+f"(d37),
          "+f"(d38),
          "+f"(d39),
          "+f"(d40),
          "+f"(d41),
          "+f"(d42),
          "+f"(d43),
          "+f"(d44),
          "+f"(d45),
          "+f"(d46),
          "+f"(d47),
          "+f"(d48),
          "+f"(d49),
          "+f"(d50),
          "+f"(d51),
          "+f"(d52),
          "+f"(d53),
          "+f"(d54),
          "+f"(d55),
          "+f"(d56),
          "+f"(d57),
          "+f"(d58),
          "+f"(d59),
          "+f"(d60),
          "+f"(d61),
          "+f"(d62),
          "+f"(d63),
          "+f"(d64),
          "+f"(d65),
          "+f"(d66),
          "+f"(d67),
          "+f"(d68),
          "+f"(d69),
          "+f"(d70),
          "+f"(d71),
          "+f"(d72),
          "+f"(d73),
          "+f"(d74),
          "+f"(d75),
          "+f"(d76),
          "+f"(d77),
          "+f"(d78),
          "+f"(d79),
          "+f"(d80),
          "+f"(d81),
          "+f"(d82),
          "+f"(d83),
          "+f"(d84),
          "+f"(d85),
          "+f"(d86),
          "+f"(d87),
          "+f"(d88),
          "+f"(d89),
          "+f"(d90),
          "+f"(d91),
          "+f"(d92),
          "+f"(d93),
          "+f"(d94),
          "+f"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x8_F32TF32TF32_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   float& d000,
                                   float& d001,
                                   float& d002,
                                   float& d003,
                                   float& d004,
                                   float& d005,
                                   float& d006,
                                   float& d007,
                                   float& d008,
                                   float& d009,
                                   float& d010,
                                   float& d011,
                                   float& d012,
                                   float& d013,
                                   float& d014,
                                   float& d015,
                                   float& d016,
                                   float& d017,
                                   float& d018,
                                   float& d019,
                                   float& d020,
                                   float& d021,
                                   float& d022,
                                   float& d023,
                                   float& d024,
                                   float& d025,
                                   float& d026,
                                   float& d027,
                                   float& d028,
                                   float& d029,
                                   float& d030,
                                   float& d031,
                                   float& d032,
                                   float& d033,
                                   float& d034,
                                   float& d035,
                                   float& d036,
                                   float& d037,
                                   float& d038,
                                   float& d039,
                                   float& d040,
                                   float& d041,
                                   float& d042,
                                   float& d043,
                                   float& d044,
                                   float& d045,
                                   float& d046,
                                   float& d047,
                                   float& d048,
                                   float& d049,
                                   float& d050,
                                   float& d051,
                                   float& d052,
                                   float& d053,
                                   float& d054,
                                   float& d055,
                                   float& d056,
                                   float& d057,
                                   float& d058,
                                   float& d059,
                                   float& d060,
                                   float& d061,
                                   float& d062,
                                   float& d063,
                                   float& d064,
                                   float& d065,
                                   float& d066,
                                   float& d067,
                                   float& d068,
                                   float& d069,
                                   float& d070,
                                   float& d071,
                                   float& d072,
                                   float& d073,
                                   float& d074,
                                   float& d075,
                                   float& d076,
                                   float& d077,
                                   float& d078,
                                   float& d079,
                                   float& d080,
                                   float& d081,
                                   float& d082,
                                   float& d083,
                                   float& d084,
                                   float& d085,
                                   float& d086,
                                   float& d087,
                                   float& d088,
                                   float& d089,
                                   float& d090,
                                   float& d091,
                                   float& d092,
                                   float& d093,
                                   float& d094,
                                   float& d095,
                                   float& d096,
                                   float& d097,
                                   float& d098,
                                   float& d099,
                                   float& d100,
                                   float& d101,
                                   float& d102,
                                   float& d103,
                                   float& d104,
                                   float& d105,
                                   float& d106,
                                   float& d107,
                                   float& d108,
                                   float& d109,
                                   float& d110,
                                   float& d111,
                                   float& d112,
                                   float& d113,
                                   float& d114,
                                   float& d115,
                                   float& d116,
                                   float& d117,
                                   float& d118,
                                   float& d119,
                                   float& d120,
                                   float& d121,
                                   float& d122,
                                   float& d123,
                                   float& d124,
                                   float& d125,
                                   float& d126,
                                   float& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130, %131, %132;\n"
        : "+f"(d000),
          "+f"(d001),
          "+f"(d002),
          "+f"(d003),
          "+f"(d004),
          "+f"(d005),
          "+f"(d006),
          "+f"(d007),
          "+f"(d008),
          "+f"(d009),
          "+f"(d010),
          "+f"(d011),
          "+f"(d012),
          "+f"(d013),
          "+f"(d014),
          "+f"(d015),
          "+f"(d016),
          "+f"(d017),
          "+f"(d018),
          "+f"(d019),
          "+f"(d020),
          "+f"(d021),
          "+f"(d022),
          "+f"(d023),
          "+f"(d024),
          "+f"(d025),
          "+f"(d026),
          "+f"(d027),
          "+f"(d028),
          "+f"(d029),
          "+f"(d030),
          "+f"(d031),
          "+f"(d032),
          "+f"(d033),
          "+f"(d034),
          "+f"(d035),
          "+f"(d036),
          "+f"(d037),
          "+f"(d038),
          "+f"(d039),
          "+f"(d040),
          "+f"(d041),
          "+f"(d042),
          "+f"(d043),
          "+f"(d044),
          "+f"(d045),
          "+f"(d046),
          "+f"(d047),
          "+f"(d048),
          "+f"(d049),
          "+f"(d050),
          "+f"(d051),
          "+f"(d052),
          "+f"(d053),
          "+f"(d054),
          "+f"(d055),
          "+f"(d056),
          "+f"(d057),
          "+f"(d058),
          "+f"(d059),
          "+f"(d060),
          "+f"(d061),
          "+f"(d062),
          "+f"(d063),
          "+f"(d064),
          "+f"(d065),
          "+f"(d066),
          "+f"(d067),
          "+f"(d068),
          "+f"(d069),
          "+f"(d070),
          "+f"(d071),
          "+f"(d072),
          "+f"(d073),
          "+f"(d074),
          "+f"(d075),
          "+f"(d076),
          "+f"(d077),
          "+f"(d078),
          "+f"(d079),
          "+f"(d080),
          "+f"(d081),
          "+f"(d082),
          "+f"(d083),
          "+f"(d084),
          "+f"(d085),
          "+f"(d086),
          "+f"(d087),
          "+f"(d088),
          "+f"(d089),
          "+f"(d090),
          "+f"(d091),
          "+f"(d092),
          "+f"(d093),
          "+f"(d094),
          "+f"(d095),
          "+f"(d096),
          "+f"(d097),
          "+f"(d098),
          "+f"(d099),
          "+f"(d100),
          "+f"(d101),
          "+f"(d102),
          "+f"(d103),
          "+f"(d104),
          "+f"(d105),
          "+f"(d106),
          "+f"(d107),
          "+f"(d108),
          "+f"(d109),
          "+f"(d110),
          "+f"(d111),
          "+f"(d112),
          "+f"(d113),
          "+f"(d114),
          "+f"(d115),
          "+f"(d116),
          "+f"(d117),
          "+f"(d118),
          "+f"(d119),
          "+f"(d120),
          "+f"(d121),
          "+f"(d122),
          "+f"(d123),
          "+f"(d124),
          "+f"(d125),
          "+f"(d126),
          "+f"(d127)
        : "l"(desc_a),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x8_F32TF32TF32_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x256x8 TN F32+=TF32*TF32
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One,
          GMMA::ScaleIn scaleA = GMMA::ScaleIn::One,
          GMMA::ScaleIn scaleB = GMMA::ScaleIn::One>
struct SM90_64x256x8_F32TF32TF32_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = float[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   float& d000,
                                   float& d001,
                                   float& d002,
                                   float& d003,
                                   float& d004,
                                   float& d005,
                                   float& d006,
                                   float& d007,
                                   float& d008,
                                   float& d009,
                                   float& d010,
                                   float& d011,
                                   float& d012,
                                   float& d013,
                                   float& d014,
                                   float& d015,
                                   float& d016,
                                   float& d017,
                                   float& d018,
                                   float& d019,
                                   float& d020,
                                   float& d021,
                                   float& d022,
                                   float& d023,
                                   float& d024,
                                   float& d025,
                                   float& d026,
                                   float& d027,
                                   float& d028,
                                   float& d029,
                                   float& d030,
                                   float& d031,
                                   float& d032,
                                   float& d033,
                                   float& d034,
                                   float& d035,
                                   float& d036,
                                   float& d037,
                                   float& d038,
                                   float& d039,
                                   float& d040,
                                   float& d041,
                                   float& d042,
                                   float& d043,
                                   float& d044,
                                   float& d045,
                                   float& d046,
                                   float& d047,
                                   float& d048,
                                   float& d049,
                                   float& d050,
                                   float& d051,
                                   float& d052,
                                   float& d053,
                                   float& d054,
                                   float& d055,
                                   float& d056,
                                   float& d057,
                                   float& d058,
                                   float& d059,
                                   float& d060,
                                   float& d061,
                                   float& d062,
                                   float& d063,
                                   float& d064,
                                   float& d065,
                                   float& d066,
                                   float& d067,
                                   float& d068,
                                   float& d069,
                                   float& d070,
                                   float& d071,
                                   float& d072,
                                   float& d073,
                                   float& d074,
                                   float& d075,
                                   float& d076,
                                   float& d077,
                                   float& d078,
                                   float& d079,
                                   float& d080,
                                   float& d081,
                                   float& d082,
                                   float& d083,
                                   float& d084,
                                   float& d085,
                                   float& d086,
                                   float& d087,
                                   float& d088,
                                   float& d089,
                                   float& d090,
                                   float& d091,
                                   float& d092,
                                   float& d093,
                                   float& d094,
                                   float& d095,
                                   float& d096,
                                   float& d097,
                                   float& d098,
                                   float& d099,
                                   float& d100,
                                   float& d101,
                                   float& d102,
                                   float& d103,
                                   float& d104,
                                   float& d105,
                                   float& d106,
                                   float& d107,
                                   float& d108,
                                   float& d109,
                                   float& d110,
                                   float& d111,
                                   float& d112,
                                   float& d113,
                                   float& d114,
                                   float& d115,
                                   float& d116,
                                   float& d117,
                                   float& d118,
                                   float& d119,
                                   float& d120,
                                   float& d121,
                                   float& d122,
                                   float& d123,
                                   float& d124,
                                   float& d125,
                                   float& d126,
                                   float& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k8.f32.tf32.tf32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133, %134, %135;\n"
        : "+f"(d000),
          "+f"(d001),
          "+f"(d002),
          "+f"(d003),
          "+f"(d004),
          "+f"(d005),
          "+f"(d006),
          "+f"(d007),
          "+f"(d008),
          "+f"(d009),
          "+f"(d010),
          "+f"(d011),
          "+f"(d012),
          "+f"(d013),
          "+f"(d014),
          "+f"(d015),
          "+f"(d016),
          "+f"(d017),
          "+f"(d018),
          "+f"(d019),
          "+f"(d020),
          "+f"(d021),
          "+f"(d022),
          "+f"(d023),
          "+f"(d024),
          "+f"(d025),
          "+f"(d026),
          "+f"(d027),
          "+f"(d028),
          "+f"(d029),
          "+f"(d030),
          "+f"(d031),
          "+f"(d032),
          "+f"(d033),
          "+f"(d034),
          "+f"(d035),
          "+f"(d036),
          "+f"(d037),
          "+f"(d038),
          "+f"(d039),
          "+f"(d040),
          "+f"(d041),
          "+f"(d042),
          "+f"(d043),
          "+f"(d044),
          "+f"(d045),
          "+f"(d046),
          "+f"(d047),
          "+f"(d048),
          "+f"(d049),
          "+f"(d050),
          "+f"(d051),
          "+f"(d052),
          "+f"(d053),
          "+f"(d054),
          "+f"(d055),
          "+f"(d056),
          "+f"(d057),
          "+f"(d058),
          "+f"(d059),
          "+f"(d060),
          "+f"(d061),
          "+f"(d062),
          "+f"(d063),
          "+f"(d064),
          "+f"(d065),
          "+f"(d066),
          "+f"(d067),
          "+f"(d068),
          "+f"(d069),
          "+f"(d070),
          "+f"(d071),
          "+f"(d072),
          "+f"(d073),
          "+f"(d074),
          "+f"(d075),
          "+f"(d076),
          "+f"(d077),
          "+f"(d078),
          "+f"(d079),
          "+f"(d080),
          "+f"(d081),
          "+f"(d082),
          "+f"(d083),
          "+f"(d084),
          "+f"(d085),
          "+f"(d086),
          "+f"(d087),
          "+f"(d088),
          "+f"(d089),
          "+f"(d090),
          "+f"(d091),
          "+f"(d092),
          "+f"(d093),
          "+f"(d094),
          "+f"(d095),
          "+f"(d096),
          "+f"(d097),
          "+f"(d098),
          "+f"(d099),
          "+f"(d100),
          "+f"(d101),
          "+f"(d102),
          "+f"(d103),
          "+f"(d104),
          "+f"(d105),
          "+f"(d106),
          "+f"(d107),
          "+f"(d108),
          "+f"(d109),
          "+f"(d110),
          "+f"(d111),
          "+f"(d112),
          "+f"(d113),
          "+f"(d114),
          "+f"(d115),
          "+f"(d116),
          "+f"(d117),
          "+f"(d118),
          "+f"(d119),
          "+f"(d120),
          "+f"(d121),
          "+f"(d122),
          "+f"(d123),
          "+f"(d124),
          "+f"(d125),
          "+f"(d126),
          "+f"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)),
          "n"(int32_t(scaleA)),
          "n"(int32_t(scaleB)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x8_F32TF32TF32_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=S8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32S8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.s8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32S8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8S8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8S8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8S8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8S8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.s8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8S8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.s8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8S8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*S8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8S8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.s8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8S8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3},"
        " %4,"
        " %5,"
        " %6;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        " %8,"
        " %9,"
        " %10;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        " %16,"
        " %17,"
        " %18;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        " %32,"
        " %33,"
        " %34;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47},"
        " %48,"
        " %49,"
        " %50;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " %66;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        " %96,"
        " %97,"
        " %98;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8U8_SS_TN {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8U8_SS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8U8_SS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " %128,"
        " %129,"
        " %130;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8U8_SS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x8x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x8x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        " %8,"
        " %9;\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x8x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x16x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x16x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[8];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a0,
                                   uint32_t const& a1,
                                   uint32_t const& a2,
                                   uint32_t const& a3,
                                   uint64_t const& desc_b,
                                   uint32_t& d0,
                                   uint32_t& d1,
                                   uint32_t& d2,
                                   uint32_t& d3,
                                   uint32_t& d4,
                                   uint32_t& d5,
                                   uint32_t& d6,
                                   uint32_t& d7) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9,  %10, %11},"
        " %12,"
        " %13;\n"
        : "+r"(d0),
          "+r"(d1),
          "+r"(d2),
          "+r"(d3),
          "+r"(d4),
          "+r"(d5),
          "+r"(d6),
          "+r"(d7)
        : "r"(a0),
          "r"(a1),
          "r"(a2),
          "r"(a3),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x16x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x32x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x32x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15},"
        "{%16, %17, %18, %19},"
        " %20,"
        " %21;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x32x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.u8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x64x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x64x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[32];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.u8.u8.satfinite "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " %37;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x64x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x96x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x96x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[48];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n96k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47},"
        "{%48,  %49,  %50,  %51},"
        " %52,"
        " %53;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x96x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x128x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x128x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[64];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        "{%64,  %65,  %66,  %67},"
        " %68,"
        " %69;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x128x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x192x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x192x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[96];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a00,
                                   uint32_t const& a01,
                                   uint32_t const& a02,
                                   uint32_t const& a03,
                                   uint64_t const& desc_b,
                                   uint32_t& d00,
                                   uint32_t& d01,
                                   uint32_t& d02,
                                   uint32_t& d03,
                                   uint32_t& d04,
                                   uint32_t& d05,
                                   uint32_t& d06,
                                   uint32_t& d07,
                                   uint32_t& d08,
                                   uint32_t& d09,
                                   uint32_t& d10,
                                   uint32_t& d11,
                                   uint32_t& d12,
                                   uint32_t& d13,
                                   uint32_t& d14,
                                   uint32_t& d15,
                                   uint32_t& d16,
                                   uint32_t& d17,
                                   uint32_t& d18,
                                   uint32_t& d19,
                                   uint32_t& d20,
                                   uint32_t& d21,
                                   uint32_t& d22,
                                   uint32_t& d23,
                                   uint32_t& d24,
                                   uint32_t& d25,
                                   uint32_t& d26,
                                   uint32_t& d27,
                                   uint32_t& d28,
                                   uint32_t& d29,
                                   uint32_t& d30,
                                   uint32_t& d31,
                                   uint32_t& d32,
                                   uint32_t& d33,
                                   uint32_t& d34,
                                   uint32_t& d35,
                                   uint32_t& d36,
                                   uint32_t& d37,
                                   uint32_t& d38,
                                   uint32_t& d39,
                                   uint32_t& d40,
                                   uint32_t& d41,
                                   uint32_t& d42,
                                   uint32_t& d43,
                                   uint32_t& d44,
                                   uint32_t& d45,
                                   uint32_t& d46,
                                   uint32_t& d47,
                                   uint32_t& d48,
                                   uint32_t& d49,
                                   uint32_t& d50,
                                   uint32_t& d51,
                                   uint32_t& d52,
                                   uint32_t& d53,
                                   uint32_t& d54,
                                   uint32_t& d55,
                                   uint32_t& d56,
                                   uint32_t& d57,
                                   uint32_t& d58,
                                   uint32_t& d59,
                                   uint32_t& d60,
                                   uint32_t& d61,
                                   uint32_t& d62,
                                   uint32_t& d63,
                                   uint32_t& d64,
                                   uint32_t& d65,
                                   uint32_t& d66,
                                   uint32_t& d67,
                                   uint32_t& d68,
                                   uint32_t& d69,
                                   uint32_t& d70,
                                   uint32_t& d71,
                                   uint32_t& d72,
                                   uint32_t& d73,
                                   uint32_t& d74,
                                   uint32_t& d75,
                                   uint32_t& d76,
                                   uint32_t& d77,
                                   uint32_t& d78,
                                   uint32_t& d79,
                                   uint32_t& d80,
                                   uint32_t& d81,
                                   uint32_t& d82,
                                   uint32_t& d83,
                                   uint32_t& d84,
                                   uint32_t& d85,
                                   uint32_t& d86,
                                   uint32_t& d87,
                                   uint32_t& d88,
                                   uint32_t& d89,
                                   uint32_t& d90,
                                   uint32_t& d91,
                                   uint32_t& d92,
                                   uint32_t& d93,
                                   uint32_t& d94,
                                   uint32_t& d95) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n192k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95},"
        "{%96,  %97,  %98,  %99},"
        " %100,"
        " %101;\n"
        : "+r"(d00),
          "+r"(d01),
          "+r"(d02),
          "+r"(d03),
          "+r"(d04),
          "+r"(d05),
          "+r"(d06),
          "+r"(d07),
          "+r"(d08),
          "+r"(d09),
          "+r"(d10),
          "+r"(d11),
          "+r"(d12),
          "+r"(d13),
          "+r"(d14),
          "+r"(d15),
          "+r"(d16),
          "+r"(d17),
          "+r"(d18),
          "+r"(d19),
          "+r"(d20),
          "+r"(d21),
          "+r"(d22),
          "+r"(d23),
          "+r"(d24),
          "+r"(d25),
          "+r"(d26),
          "+r"(d27),
          "+r"(d28),
          "+r"(d29),
          "+r"(d30),
          "+r"(d31),
          "+r"(d32),
          "+r"(d33),
          "+r"(d34),
          "+r"(d35),
          "+r"(d36),
          "+r"(d37),
          "+r"(d38),
          "+r"(d39),
          "+r"(d40),
          "+r"(d41),
          "+r"(d42),
          "+r"(d43),
          "+r"(d44),
          "+r"(d45),
          "+r"(d46),
          "+r"(d47),
          "+r"(d48),
          "+r"(d49),
          "+r"(d50),
          "+r"(d51),
          "+r"(d52),
          "+r"(d53),
          "+r"(d54),
          "+r"(d55),
          "+r"(d56),
          "+r"(d57),
          "+r"(d58),
          "+r"(d59),
          "+r"(d60),
          "+r"(d61),
          "+r"(d62),
          "+r"(d63),
          "+r"(d64),
          "+r"(d65),
          "+r"(d66),
          "+r"(d67),
          "+r"(d68),
          "+r"(d69),
          "+r"(d70),
          "+r"(d71),
          "+r"(d72),
          "+r"(d73),
          "+r"(d74),
          "+r"(d75),
          "+r"(d76),
          "+r"(d77),
          "+r"(d78),
          "+r"(d79),
          "+r"(d80),
          "+r"(d81),
          "+r"(d82),
          "+r"(d83),
          "+r"(d84),
          "+r"(d85),
          "+r"(d86),
          "+r"(d87),
          "+r"(d88),
          "+r"(d89),
          "+r"(d90),
          "+r"(d91),
          "+r"(d92),
          "+r"(d93),
          "+r"(d94),
          "+r"(d95)
        : "r"(a00),
          "r"(a01),
          "r"(a02),
          "r"(a03),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x192x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8U8_RS_TN {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.u8 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8U8_RS_TN without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// MMA 64x256x32 TN S32+=U8*U8
template <GMMA::ScaleOut scaleD = GMMA::ScaleOut::One>
struct SM90_64x256x32_S32U8U8_RS_TN_SATURATE {
  using DRegisters = void;
  using ARegisters = uint32_t[4];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[128];

  CUTE_HOST_DEVICE static void fma(uint32_t const& a000,
                                   uint32_t const& a001,
                                   uint32_t const& a002,
                                   uint32_t const& a003,
                                   uint64_t const& desc_b,
                                   uint32_t& d000,
                                   uint32_t& d001,
                                   uint32_t& d002,
                                   uint32_t& d003,
                                   uint32_t& d004,
                                   uint32_t& d005,
                                   uint32_t& d006,
                                   uint32_t& d007,
                                   uint32_t& d008,
                                   uint32_t& d009,
                                   uint32_t& d010,
                                   uint32_t& d011,
                                   uint32_t& d012,
                                   uint32_t& d013,
                                   uint32_t& d014,
                                   uint32_t& d015,
                                   uint32_t& d016,
                                   uint32_t& d017,
                                   uint32_t& d018,
                                   uint32_t& d019,
                                   uint32_t& d020,
                                   uint32_t& d021,
                                   uint32_t& d022,
                                   uint32_t& d023,
                                   uint32_t& d024,
                                   uint32_t& d025,
                                   uint32_t& d026,
                                   uint32_t& d027,
                                   uint32_t& d028,
                                   uint32_t& d029,
                                   uint32_t& d030,
                                   uint32_t& d031,
                                   uint32_t& d032,
                                   uint32_t& d033,
                                   uint32_t& d034,
                                   uint32_t& d035,
                                   uint32_t& d036,
                                   uint32_t& d037,
                                   uint32_t& d038,
                                   uint32_t& d039,
                                   uint32_t& d040,
                                   uint32_t& d041,
                                   uint32_t& d042,
                                   uint32_t& d043,
                                   uint32_t& d044,
                                   uint32_t& d045,
                                   uint32_t& d046,
                                   uint32_t& d047,
                                   uint32_t& d048,
                                   uint32_t& d049,
                                   uint32_t& d050,
                                   uint32_t& d051,
                                   uint32_t& d052,
                                   uint32_t& d053,
                                   uint32_t& d054,
                                   uint32_t& d055,
                                   uint32_t& d056,
                                   uint32_t& d057,
                                   uint32_t& d058,
                                   uint32_t& d059,
                                   uint32_t& d060,
                                   uint32_t& d061,
                                   uint32_t& d062,
                                   uint32_t& d063,
                                   uint32_t& d064,
                                   uint32_t& d065,
                                   uint32_t& d066,
                                   uint32_t& d067,
                                   uint32_t& d068,
                                   uint32_t& d069,
                                   uint32_t& d070,
                                   uint32_t& d071,
                                   uint32_t& d072,
                                   uint32_t& d073,
                                   uint32_t& d074,
                                   uint32_t& d075,
                                   uint32_t& d076,
                                   uint32_t& d077,
                                   uint32_t& d078,
                                   uint32_t& d079,
                                   uint32_t& d080,
                                   uint32_t& d081,
                                   uint32_t& d082,
                                   uint32_t& d083,
                                   uint32_t& d084,
                                   uint32_t& d085,
                                   uint32_t& d086,
                                   uint32_t& d087,
                                   uint32_t& d088,
                                   uint32_t& d089,
                                   uint32_t& d090,
                                   uint32_t& d091,
                                   uint32_t& d092,
                                   uint32_t& d093,
                                   uint32_t& d094,
                                   uint32_t& d095,
                                   uint32_t& d096,
                                   uint32_t& d097,
                                   uint32_t& d098,
                                   uint32_t& d099,
                                   uint32_t& d100,
                                   uint32_t& d101,
                                   uint32_t& d102,
                                   uint32_t& d103,
                                   uint32_t& d104,
                                   uint32_t& d105,
                                   uint32_t& d106,
                                   uint32_t& d107,
                                   uint32_t& d108,
                                   uint32_t& d109,
                                   uint32_t& d110,
                                   uint32_t& d111,
                                   uint32_t& d112,
                                   uint32_t& d113,
                                   uint32_t& d114,
                                   uint32_t& d115,
                                   uint32_t& d116,
                                   uint32_t& d117,
                                   uint32_t& d118,
                                   uint32_t& d119,
                                   uint32_t& d120,
                                   uint32_t& d121,
                                   uint32_t& d122,
                                   uint32_t& d123,
                                   uint32_t& d124,
                                   uint32_t& d125,
                                   uint32_t& d126,
                                   uint32_t& d127) {
#if defined(CUTE_ARCH_MMA_SM90_ENABLED)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.s32.u8.u8.satfinite "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
        " %104, %105, %106, %107, %108, %109, %110, %111, "
        " %112, %113, %114, %115, %116, %117, %118, %119, "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        "{%128, %129, %130, %131},"
        " %132,"
        " %133;\n"
        : "+r"(d000),
          "+r"(d001),
          "+r"(d002),
          "+r"(d003),
          "+r"(d004),
          "+r"(d005),
          "+r"(d006),
          "+r"(d007),
          "+r"(d008),
          "+r"(d009),
          "+r"(d010),
          "+r"(d011),
          "+r"(d012),
          "+r"(d013),
          "+r"(d014),
          "+r"(d015),
          "+r"(d016),
          "+r"(d017),
          "+r"(d018),
          "+r"(d019),
          "+r"(d020),
          "+r"(d021),
          "+r"(d022),
          "+r"(d023),
          "+r"(d024),
          "+r"(d025),
          "+r"(d026),
          "+r"(d027),
          "+r"(d028),
          "+r"(d029),
          "+r"(d030),
          "+r"(d031),
          "+r"(d032),
          "+r"(d033),
          "+r"(d034),
          "+r"(d035),
          "+r"(d036),
          "+r"(d037),
          "+r"(d038),
          "+r"(d039),
          "+r"(d040),
          "+r"(d041),
          "+r"(d042),
          "+r"(d043),
          "+r"(d044),
          "+r"(d045),
          "+r"(d046),
          "+r"(d047),
          "+r"(d048),
          "+r"(d049),
          "+r"(d050),
          "+r"(d051),
          "+r"(d052),
          "+r"(d053),
          "+r"(d054),
          "+r"(d055),
          "+r"(d056),
          "+r"(d057),
          "+r"(d058),
          "+r"(d059),
          "+r"(d060),
          "+r"(d061),
          "+r"(d062),
          "+r"(d063),
          "+r"(d064),
          "+r"(d065),
          "+r"(d066),
          "+r"(d067),
          "+r"(d068),
          "+r"(d069),
          "+r"(d070),
          "+r"(d071),
          "+r"(d072),
          "+r"(d073),
          "+r"(d074),
          "+r"(d075),
          "+r"(d076),
          "+r"(d077),
          "+r"(d078),
          "+r"(d079),
          "+r"(d080),
          "+r"(d081),
          "+r"(d082),
          "+r"(d083),
          "+r"(d084),
          "+r"(d085),
          "+r"(d086),
          "+r"(d087),
          "+r"(d088),
          "+r"(d089),
          "+r"(d090),
          "+r"(d091),
          "+r"(d092),
          "+r"(d093),
          "+r"(d094),
          "+r"(d095),
          "+r"(d096),
          "+r"(d097),
          "+r"(d098),
          "+r"(d099),
          "+r"(d100),
          "+r"(d101),
          "+r"(d102),
          "+r"(d103),
          "+r"(d104),
          "+r"(d105),
          "+r"(d106),
          "+r"(d107),
          "+r"(d108),
          "+r"(d109),
          "+r"(d110),
          "+r"(d111),
          "+r"(d112),
          "+r"(d113),
          "+r"(d114),
          "+r"(d115),
          "+r"(d116),
          "+r"(d117),
          "+r"(d118),
          "+r"(d119),
          "+r"(d120),
          "+r"(d121),
          "+r"(d122),
          "+r"(d123),
          "+r"(d124),
          "+r"(d125),
          "+r"(d126),
          "+r"(d127)
        : "r"(a000),
          "r"(a001),
          "r"(a002),
          "r"(a003),
          "l"(desc_b),
          "n"(int32_t(scaleD)));
#else
    CUTE_RUNTIME_ASSERT(
        "Attempting to use SM90_64x256x32_S32U8U8_RS_TN_SATURATE without "
        "CUTE_ARCH_MMA_SM90_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cute
