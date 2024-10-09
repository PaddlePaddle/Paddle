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

#include <limits>

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstdint>
#else
#include <cstdint>
#endif

#include <cute/config.hpp>

namespace cute {

//
// Common Operations
//

template <class T,
          class U,
          __CUTE_REQUIRES(
              std::is_arithmetic<T>::value&& std::is_arithmetic<U>::value)>
CUTE_HOST_DEVICE constexpr auto max(T const& t, U const& u) {
  return t < u ? u : t;
}

template <class T,
          class U,
          __CUTE_REQUIRES(
              std::is_arithmetic<T>::value&& std::is_arithmetic<U>::value)>
CUTE_HOST_DEVICE constexpr auto min(T const& t, U const& u) {
  return t < u ? t : u;
}

template <class T, __CUTE_REQUIRES(std::is_arithmetic<T>::value)>
CUTE_HOST_DEVICE constexpr auto abs(T const& t) {
  if constexpr (std::is_signed<T>::value) {
    return t < T(0) ? -t : t;
  } else {
    return t;
  }

  CUTE_GCC_UNREACHABLE;
}

//
// C++17 <numeric> operations
//

// Greatest common divisor of two integers
template <
    class T,
    class U,
    __CUTE_REQUIRES(std::is_integral<T>::value&& std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr auto gcd(T t, U u) {
  while (true) {
    if (t == 0) {
      return u;
    }
    u %= t;
    if (u == 0) {
      return t;
    }
    t %= u;
  }
}

// Least common multiple of two integers
template <
    class T,
    class U,
    __CUTE_REQUIRES(std::is_integral<T>::value&& std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr auto lcm(T const& t, U const& u) {
  return (t / gcd(t, u)) * u;
}

//
// C++20 <bit> operations
//

// Checks if a number is an integral power of two
template <class T>
CUTE_HOST_DEVICE constexpr bool has_single_bit(T x) {
  return x != 0 && (x & (x - 1)) == 0;
}

// Smallest number of bits needed to represent the given value
// bit_width( 0b0000 ) = 0
// bit_width( 0b0001 ) = 1
// bit_width( 0b0010 ) = 2
// bit_width( 0b0011 ) = 2
// bit_width( 0b0100 ) = 3
// bit_width( 0b0101 ) = 3
// bit_width( 0b0110 ) = 3
// bit_width( 0b0111 ) = 3
template <class T>
CUTE_HOST_DEVICE constexpr T bit_width(T x) {
  static_assert(std::is_unsigned<T>::value,
                "Only to be used for unsigned types.");
  constexpr int N = (std::numeric_limits<T>::digits == 64
                         ? 6
                         : (std::numeric_limits<T>::digits == 32
                                ? 5
                                : (std::numeric_limits<T>::digits == 16
                                       ? 4
                                       : (std::numeric_limits<T>::digits == 8
                                              ? 3
                                              : (assert(false), 0)))));
  T r = 0;
  for (int i = N - 1; i >= 0; --i) {
    T shift = (x > ((T(1) << (T(1) << i)) - 1)) << i;
    x >>= shift;
    r |= shift;
  }
  return r + (x != 0);
}

// Smallest integral power of two not less than the given value
// bit_ceil( 0b00000000 ) = 0b00000001
// bit_ceil( 0b00000001 ) = 0b00000001
// bit_ceil( 0b00000010 ) = 0b00000010
// bit_ceil( 0b00000011 ) = 0b00000100
// bit_ceil( 0b00000100 ) = 0b00000100
// bit_ceil( 0b00000101 ) = 0b00001000
// bit_ceil( 0b00000110 ) = 0b00001000
// bit_ceil( 0b00000111 ) = 0b00001000
// bit_ceil( 0b00001000 ) = 0b00001000
// bit_ceil( 0b00001001 ) = 0b00010000
template <class T>
CUTE_HOST_DEVICE constexpr T bit_ceil(T x) {
  return x == 0 ? T(1) : (T(1) << bit_width(x - 1));
}

// Largest integral power of two not greater than the given value
// bit_floor( 0b00000000 ) = 0b00000000
// bit_floor( 0b00000001 ) = 0b00000001
// bit_floor( 0b00000010 ) = 0b00000010
// bit_floor( 0b00000011 ) = 0b00000010
// bit_floor( 0b00000100 ) = 0b00000100
// bit_floor( 0b00000101 ) = 0b00000100
// bit_floor( 0b00000110 ) = 0b00000100
// bit_floor( 0b00000111 ) = 0b00000100
// bit_floor( 0b00001000 ) = 0b00001000
// bit_floor( 0b00001001 ) = 0b00001000
template <class T>
CUTE_HOST_DEVICE constexpr T bit_floor(T x) {
  return x == 0 ? 0 : (T(1) << (bit_width(x) - 1));
}

template <class T>
CUTE_HOST_DEVICE constexpr T rotl(T x, int s);
template <class T>
CUTE_HOST_DEVICE constexpr T rotr(T x, int s);

// Computes the result of circular bitwise left-rotation
template <class T>
CUTE_HOST_DEVICE constexpr T rotl(T x, int s) {
  constexpr int N = std::numeric_limits<T>::digits;
  return s == 0 ? x : s > 0 ? (x << s) | (x >> (N - s)) : rotr(x, -s);
}

// Computes the result of circular bitwise right-rotation
template <class T>
CUTE_HOST_DEVICE constexpr T rotr(T x, int s) {
  constexpr int N = std::numeric_limits<T>::digits;
  return s == 0 ? x : s > 0 ? (x >> s) | (x << (N - s)) : rotl(x, -s);
}

// Counts the number of consecutive 0 bits, starting from the most significant
// bit countl_zero( 0b00000000 ) = 8 countl_zero( 0b11111111 ) = 0 countl_zero(
// 0b00011100 ) = 3
template <class T>
CUTE_HOST_DEVICE constexpr T countl_zero(T x) {
  return std::numeric_limits<T>::digits - bit_width(x);
}

// Counts the number of consecutive 1 bits, starting from the most significant
// bit countl_one( 0b00000000 ) = 0 countl_one( 0b11111111 ) = 8 countl_one(
// 0b11100011 ) = 3
template <class T>
CUTE_HOST_DEVICE constexpr T countl_one(T x) {
  return countl_zero(~x);
}

// Counts the number of consecutive 0 bits, starting from the least significant
// bit countr_zero( 0b00000000 ) = 8 countr_zero( 0b11111111 ) = 0 countr_zero(
// 0b00011100 ) = 2
template <class T>
CUTE_HOST_DEVICE constexpr T countr_zero(T x) {
  return x == 0 ? std::numeric_limits<T>::digits
                : bit_width(T(x & T(-x))) - 1;  // bit_width of the LSB
}

// Counts the number of consecutive 1 bits, starting from the least significant
// bit countr_one( 0b00000000 ) = 0 countr_one( 0b11111111 ) = 8 countr_one(
// 0b11100011 ) = 2
template <class T>
CUTE_HOST_DEVICE constexpr T countr_one(T x) {
  return countr_zero(~x);
}

// Counts the number of 1 bits in an unsigned integer
// popcount( 0b00000000 ) = 0
// popcount( 0b11111111 ) = 8
// popcount( 0b00011101 ) = 4
template <class T>
CUTE_HOST_DEVICE constexpr int popcount(T x) {
  int c = 0;
  while (x) {
    ++c;
    x &= x - 1;  // clear the least significant bit set
  }
  return c;
}

//
// Custom operations
//

// Computes the result of bitwise left-shift
template <class T>
CUTE_HOST_DEVICE constexpr T shiftl(T x, int s) {
  return s >= 0 ? (x << s) : (x >> -s);
}

// Computes the result of bitwise right-shift
template <class T>
CUTE_HOST_DEVICE constexpr T shiftr(T x, int s) {
  return s >= 0 ? (x >> s) : (x << -s);
}

// Returns 1 if x > 0, -1 if x < 0, and 0 if x is zero.
template <class T, __CUTE_REQUIRES(std::is_unsigned<T>::value)>
CUTE_HOST_DEVICE constexpr int signum(T const& x) {
  return T(0) < x;
}

template <class T, __CUTE_REQUIRES(not std::is_unsigned<T>::value)>
CUTE_HOST_DEVICE constexpr int signum(T const& x) {
  return (T(0) < x) - (x < T(0));
}

// Safe divide
// @pre t % u == 0
// @result t / u
template <
    class T,
    class U,
    __CUTE_REQUIRES(std::is_integral<T>::value&& std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr auto safe_div(T const& t, U const& u) {
  // assert(t % u == 0);
  return t / u;
}

}  // namespace cute
