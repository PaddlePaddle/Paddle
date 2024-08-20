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

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#endif

#include <cute/config.hpp>

/// Optionally enable GCC's built-in type
#if defined(__x86_64) && !defined(__CUDA_ARCH__)
#if defined(__GNUC__) && 0
#define CUTE_UINT128_NATIVE
#elif defined(_MSC_VER)
#define CUTE_INT128_ARITHMETIC
#include <intrin.h>
#endif
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cute {

/////////////////////////////////////////////////////////////////////////////////////////////////

///! Unsigned 128b integer type
struct alignas(16) uint128_t {
  /// Size of one part of the uint's storage in bits
  static constexpr int storage_bits_ = 64;

  struct hilo {
    uint64_t lo;
    uint64_t hi;
  };

  // Use a union to store either low and high parts or, if present, a built-in
  // 128b integer type.
  union {
    struct hilo hilo_;

#if defined(CUTE_UINT128_NATIVE)
    unsigned __int128 native;
#endif  // defined(CUTE_UINT128_NATIVE)
  };

  //
  // Methods
  //

  /// Default ctor
  CUTE_HOST_DEVICE constexpr uint128_t() : hilo_{0, 0} {}

  /// Constructor from uint64
  CUTE_HOST_DEVICE constexpr uint128_t(uint64_t lo_) : hilo_{lo_, 0} {}

  /// Constructor from two 64b unsigned integers
  CUTE_HOST_DEVICE constexpr uint128_t(uint64_t lo_, uint64_t hi_)
      : hilo_{lo_, hi_} {}

  /// Optional constructor from native value
#if defined(CUTE_UINT128_NATIVE)
  uint128_t(unsigned __int128 value) : native(value) {}
#endif

  /// Lossily cast to uint64
  CUTE_HOST_DEVICE constexpr explicit operator uint64_t() const {
    return hilo_.lo;
  }

  template <class Dummy = bool>
  CUTE_HOST_DEVICE constexpr static void exception() {
    // static_assert(sizeof(Dummy) == 0, "Not implemented exception!");
    // abort();
    // printf("uint128 not implemented!\n");
  }

  /// Add
  CUTE_HOST_DEVICE constexpr uint128_t operator+(uint128_t const& rhs) const {
    uint128_t y;
#if defined(CUTE_UINT128_NATIVE)
    y.native = native + rhs.native;
#else
    y.hilo_.lo = hilo_.lo + rhs.hilo_.lo;
    y.hilo_.hi = hilo_.hi + rhs.hilo_.hi + (!y.hilo_.lo && (rhs.hilo_.lo));
#endif
    return y;
  }

  /// Subtract
  CUTE_HOST_DEVICE constexpr uint128_t operator-(uint128_t const& rhs) const {
    uint128_t y;
#if defined(CUTE_UINT128_NATIVE)
    y.native = native - rhs.native;
#else
    y.hilo_.lo = hilo_.lo - rhs.hilo_.lo;
    y.hilo_.hi =
        hilo_.hi - rhs.hilo_.hi - (rhs.hilo_.lo && y.hilo_.lo > hilo_.lo);
#endif
    return y;
  }

  /// Multiply by unsigned 64b integer yielding 128b integer
  CUTE_HOST_DEVICE constexpr uint128_t operator*(uint64_t const& rhs) const {
    uint128_t y;
#if defined(CUTE_UINT128_NATIVE)
    y.native = native * rhs;
#elif defined(CUTE_INT128_ARITHMETIC)
    // Multiply by the low part
    y.hilo_.lo = _umul128(hilo_.lo, rhs, &y.hilo_.hi);

    // Add the high part and ignore the overflow
    uint64_t overflow;
    y.hilo_.hi += _umul128(hilo_.hi, rhs, &overflow);
#else
    exception();
#endif
    return y;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  CUTE_HOST_DEVICE constexpr uint64_t operator/(uint64_t const& divisor) const {
    uint64_t quotient = 0;
#if defined(CUTE_UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
#elif defined(CUTE_INT128_ARITHMETIC)
    // implemented using MSVC's arithmetic intrinsics
    uint64_t remainder = 0;
    quotient = _udiv128(hilo_.hi, hilo_.lo, divisor, &remainder);
#else
    exception();
#endif
    return quotient;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  CUTE_HOST_DEVICE constexpr uint64_t operator%(uint64_t const& divisor) const {
    uint64_t remainder = 0;
#if defined(CUTE_UINT128_NATIVE)
    remainder = uint64_t(native % divisor);
#elif defined(CUTE_INT128_ARITHMETIC)
    // implemented using MSVC's arithmetic intrinsics
    (void)_udiv128(hilo_.hi, hilo_.lo, divisor, &remainder);
#else
    exception();
#endif
    return remainder;
  }

  /// Computes the quotient and remainder in a single method.
  CUTE_HOST_DEVICE constexpr uint64_t divmod(uint64_t& remainder,
                                             uint64_t divisor) const {
    uint64_t quotient = 0;
#if defined(CUTE_UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
    remainder = uint64_t(native % divisor);
#elif defined(CUTE_INT128_ARITHMETIC)
    // implemented using MSVC's arithmetic intrinsics
    quotient = _udiv128(hilo_.hi, hilo_.lo, divisor, &remainder);
#else
    exception();
#endif
    return quotient;
  }

  /// Left-shifts a 128b unsigned integer
  CUTE_HOST_DEVICE constexpr uint128_t operator<<(int sh) const {
    if (sh == 0) {
      return *this;
    } else if (sh >= storage_bits_) {
      return uint128_t(0, hilo_.lo << (sh - storage_bits_));
    } else {
      return uint128_t(
          (hilo_.lo << sh),
          (hilo_.hi << sh) | uint64_t(hilo_.lo >> (storage_bits_ - sh)));
    }
  }

  /// Right-shifts a 128b unsigned integer
  CUTE_HOST_DEVICE constexpr uint128_t operator>>(int sh) const {
    if (sh == 0) {
      return *this;
    } else if (sh >= storage_bits_) {
      return uint128_t((hilo_.hi >> (sh - storage_bits_)), 0);
    } else {
      return uint128_t((hilo_.lo >> sh) | (hilo_.hi << (storage_bits_ - sh)),
                       (hilo_.hi >> sh));
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cute

/////////////////////////////////////////////////////////////////////////////////////////////////
