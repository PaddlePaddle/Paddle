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
/*!
    \file
    \brief Defines a class for using integer types smaller than one byte in host or
      device code.
*/
#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#include "cutlass/platform/platform.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 4-bit signed integer type
template <int Bits, bool Signed = true>
struct integer_subbyte {

  /// Number of bits
  static int const kBits = Bits;

  /// Whether type is signed
  static bool const kSigned = Signed;

  /// External type
  using T = typename platform::conditional<kSigned, int, unsigned>::type;

  /// Storage type
  using Storage = uint8_t;

  /// Bitmask used to truncate from larger integers
  static Storage const kMask = Storage((1 << kBits) - 1);

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  CUTLASS_HOST_DEVICE
  integer_subbyte() { }

  /// Conversion from integer type
  CUTLASS_HOST_DEVICE
  integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const &>(value) & kMask) {}

  CUTLASS_HOST_DEVICE
  integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const &>(value) & kMask) {}

  CUTLASS_HOST_DEVICE
  integer_subbyte(double value) {
    T tmp = static_cast<T>(value);
    storage = Storage(reinterpret_cast<unsigned const &>(tmp) & kMask);
  }

  ///
  CUTLASS_HOST_DEVICE
  operator T() const {
    if (kSigned) {
      // Sign extend
      if (storage & Storage(1 << (kBits - 1))) {
        return T(storage) | ~T(kMask);
      }
    }
    return T(storage);
  }

  /// Equality
  CUTLASS_HOST_DEVICE
  bool operator==(integer_subbyte const &rhs) const {
    return storage == rhs.storage;
  }

  /// Inequality
  CUTLASS_HOST_DEVICE
  bool operator!=(integer_subbyte const &rhs) const {
    return storage != rhs.storage;
  }

  /// Less than or equal
  CUTLASS_HOST_DEVICE
  bool operator<=(integer_subbyte const &rhs) const {
    if (kSigned) {
      if (storage & (1 << (kBits - 1))) {
        return !(rhs.storage < storage);
      }
    }
    return storage < rhs.storage;
  }

  /// Less than
  CUTLASS_HOST_DEVICE
  bool operator<(integer_subbyte const &rhs) const {
    if (kSigned) {
      if (storage & (1 << (kBits - 1))) {
        return !(rhs.storage <= storage);
      }
    }
    return storage < rhs.storage;
  }

  /// Greater than or equal
  CUTLASS_HOST_DEVICE
  bool operator>=(integer_subbyte const &rhs) const {
    return !(*this < rhs);
  }

  /// Greater than
  CUTLASS_HOST_DEVICE
  bool operator>(integer_subbyte const &rhs) const {
    return !(*this <= rhs);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////


/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits - specialized for uint1b_t
template <>
struct sizeof_bits<uint1b_t> {
  static int const value = 1;
};

/// Defines the size of an element in bits - specialized for int2b_t
template <>
struct sizeof_bits<int2b_t> {
  static int const value = 2;
};

/// Defines the size of an element in bits - specialized for uint2b_t
template <>
struct sizeof_bits<uint2b_t> {
  static int const value = 2;
};

/// Defines the size of an element in bits - specialized for int4b_t
template <>
struct sizeof_bits<int4b_t> {
  static int const value = 4;
};

/// Defines the size of an element in bits - specialized for uint4b_t
template <>
struct sizeof_bits<uint4b_t> {
  static int const value = 4;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace platform {

template <>
struct numeric_limits<cutlass::int4b_t> {
  CUTLASS_HOST_DEVICE
  static cutlass::int4b_t const lowest() noexcept { return -8;}
  CUTLASS_HOST_DEVICE
  static cutlass::int4b_t const max() noexcept { return 7;}
  static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<cutlass::uint4b_t> {
  CUTLASS_HOST_DEVICE
  static cutlass::uint4b_t const lowest() noexcept { return 0;}
  CUTLASS_HOST_DEVICE
  static cutlass::uint4b_t const max() noexcept { return 15;}
  static constexpr bool is_integer = true;
};

template <>
struct numeric_limits<cutlass::uint1b_t> {
  CUTLASS_HOST_DEVICE
  static cutlass::uint1b_t const lowest() noexcept { return 0;}
  CUTLASS_HOST_DEVICE
  static cutlass::uint1b_t const max() noexcept { return 1;}
  static constexpr bool is_integer = true;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace platform
} // namespace cutlass
