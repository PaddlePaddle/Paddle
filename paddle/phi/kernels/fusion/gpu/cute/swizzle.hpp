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

#include <cute/config.hpp>

#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/container/tuple.hpp>
#include <cute/numeric/integer_sequence.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/math.hpp>

namespace cute {

// A generic Swizzle functor
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to
 * keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY
 * mask (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
 */
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle {
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  static_assert(num_base >= 0, "MBase must be positive.");
  static_assert(num_bits >= 0, "BBits must be positive.");
  static_assert(abs(num_shft) >= num_bits,
                "abs(SShift) must be more than BBits.");

  // using 'int' type here to avoid unintentially casting to unsigned... unsure.
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk =
      cute::constant<int, bit_msk{} << (num_base + max(0, num_shft))>;
  using zzz_msk =
      cute::constant<int, bit_msk{} << (num_base - min(0, num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk{} | zzz_msk{});

  template <class Offset, __CUTE_REQUIRES(is_integral<Offset>::value)>
  CUTE_HOST_DEVICE constexpr static auto apply(Offset const& offset) {
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});  // ZZZ ^= YYY
  }

  template <class Offset, __CUTE_REQUIRES(is_integral<Offset>::value)>
  CUTE_HOST_DEVICE constexpr auto operator()(Offset const& offset) const {
    return apply(offset);
  }
};

// Translation for legacy SwizzleXor
// TODO: Deprecate
template <uint32_t BBits, uint32_t MBase, uint32_t SShift = 0>
using SwizzleXor = Swizzle<BBits, MBase, SShift + BBits>;

//
// make_swizzle<0b1000, 0b0100>()         ->  Swizzle<1,2,1>
// make_swizzle<0b11000000, 0b00000110>() ->  Swizzle<2,1,5>
//

template <uint32_t Y, uint32_t Z>
CUTE_HOST_DEVICE constexpr auto make_swizzle() {
  constexpr uint32_t BZ = popcount(Y);  // Number of swizzle bits
  constexpr uint32_t BY = popcount(Z);  // Number of swizzle bits
  static_assert(BZ == BY, "Number of bits in Y and Z don't match");
  constexpr uint32_t TZ_Y = countr_zero(Y);  // Number of trailing zeros in Y
  constexpr uint32_t TZ_Z = countr_zero(Z);  // Number of trailing zeros in Z
  constexpr uint32_t M = cute::min(TZ_Y, TZ_Z) % 32;
  constexpr int32_t S =
      int32_t(TZ_Y) - int32_t(TZ_Z);  // Difference in trailing zeros
  static_assert((Y | Z) == Swizzle<BZ, M, S>::swizzle_code,
                "Something went wrong.");
  return Swizzle<BZ, M, S>{};
}

template <int B0, int M0, int S0, int B1, int M1, int S1>
CUTE_HOST_DEVICE constexpr auto composition(Swizzle<B0, M0, S0>,
                                            Swizzle<B1, M1, S1>) {
  static_assert(S0 == S1, "Can only merge swizzles of the same shift.");
  constexpr uint32_t Y =
      Swizzle<B0, M0, S0>::yyy_msk::value ^ Swizzle<B1, M1, S1>::yyy_msk::value;
  constexpr uint32_t Z =
      Swizzle<B0, M0, S0>::zzz_msk::value ^ Swizzle<B1, M1, S1>::zzz_msk::value;
  return make_swizzle<Y, Z>();

  // return ComposedFn<Swizzle<B0,M0,S0>, Swizzle<B1,M1,S1>>{};
}

//
// Upcast and Downcast
//

template <int N, int B, int M, int S>
CUTE_HOST_DEVICE constexpr auto upcast(Swizzle<B, M, S> const& swizzle) {
  static_assert(has_single_bit(N), "N must be a power of two");
  constexpr int log2_n = bit_width(uint32_t(N)) - 1;
  constexpr int NewM = M - log2_n;
  if constexpr (NewM >= 0) {
    return Swizzle<B, NewM, S>{};
  } else {
    return Swizzle<cute::max(B + NewM, 0), 0, S>{};
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, int B, int M, int S>
CUTE_HOST_DEVICE constexpr auto downcast(Swizzle<B, M, S> const& swizzle) {
  static_assert(has_single_bit(N), "N must be a power of two");
  constexpr int log2_n = bit_width(uint32_t(N)) - 1;
  return Swizzle<B, (M + log2_n), S>{};
}

template <class OldType, class NewType, int B, int M, int S>
CUTE_HOST_DEVICE constexpr auto recast(Swizzle<B, M, S> const& swizzle) {
  if constexpr (sizeof_bits<NewType>::value == sizeof_bits<OldType>::value) {
    return swizzle;
  } else if constexpr (sizeof_bits<NewType>::value >
                       sizeof_bits<OldType>::value) {
    static_assert(
        sizeof_bits<NewType>::value % sizeof_bits<OldType>::value == 0,
        "NewType must be a multiple of OldType");
    return upcast<sizeof_bits<NewType>::value / sizeof_bits<OldType>::value>(
        swizzle);
  } else if constexpr (sizeof_bits<NewType>::value <
                       sizeof_bits<OldType>::value) {
    static_assert(
        sizeof_bits<OldType>::value % sizeof_bits<NewType>::value == 0,
        "NewType must be a divisor of OldType");
    return downcast<sizeof_bits<OldType>::value / sizeof_bits<NewType>::value>(
        swizzle);
  }
}

//
// Utility for slicing and swizzle "offsets"
//

// For swizzle functions, it is often needed to keep track of which bits are
//   consumed and which bits are free. Furthermore, it is useful to know whether
// each of these bits is known statically or dynamically.

// MixedBits is an integer class where some bits are known statically and some
//   bits are known dynamically. These sets of bits are disjoint and it is known
//   statically which bits are known dynamically.

// MixedBits can only be manipulated through bitwise operations

// Abstract value:  StaticInt | (dynamic_int_ & StaticFlags)
template <uint32_t StaticInt = 0,
          class DynamicType = uint32_t,
          uint32_t StaticFlags = 0>  // 0: static, 1: dynamic
struct MixedBits {
  // Representation invariants
  static_assert(StaticFlags != 0,
                "Should be at least one dynamic bit in MixedBits.");
  static_assert((StaticInt & StaticFlags) == 0,
                "No static/dynamic overlap allowed in MixedBits.");
  // assert((dynamic_int_ & ~F) == 0);

  DynamicType dynamic_int_;
};

template <class S, S s, class DynamicType, class F, F f>
CUTE_HOST_DEVICE constexpr auto make_mixed_bits(constant<S, s> const&,
                                                DynamicType const& d,
                                                constant<F, f> const&) {
  static_assert(is_integral<DynamicType>::value);
  if constexpr (is_static<DynamicType>::value) {
    static_assert((s & DynamicType::value & f) == 0,
                  "No static/dynamic overlap allowed.");
    return constant<S, s>{} |
           (d & constant<F, f>{});  // Just return a static int
  } else if constexpr (f == 0) {
    return constant<S, s>{};  // Just return a static int
  } else {
    return MixedBits<s, DynamicType, f>{d & f};  // MixedBits
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Explicit conversion for now -- consider casting on plus or minus
//

template <uint32_t S, class D, uint32_t F>
CUTE_HOST_DEVICE constexpr auto to_integral(MixedBits<S, D, F> const& m) {
  // return S | (m.dynamic_int_ & F);
  return S | m.dynamic_int_;
}

// Any cute::is_integral
template <class I, __CUTE_REQUIRES(cute::is_integral<I>::value)>
CUTE_HOST_DEVICE constexpr auto to_integral(I const& i) {
  return i;
}

//
// Operators
//

// Equality
template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator==(MixedBits<S0, D0, F0> const& m,
                                           constant<TS1, S1> const&) {
  return (S0 == (S1 & ~F0)) && (m.dynamic_int_ == (S1 & F0));
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator==(constant<TS1, S1> const& s,
                                           MixedBits<S0, D0, F0> const& m) {
  return m == s;
}

// Bitwise AND
template <uint32_t S0,
          class D0,
          uint32_t F0,
          uint32_t S1,
          class D1,
          uint32_t F1>
CUTE_HOST_DEVICE constexpr auto operator&(MixedBits<S0, D0, F0> const& m0,
                                          MixedBits<S1, D1, F1> const& m1) {
  // Truth table for (S0,D0,F0) & (S1,D1,F1) -> (S,D,F)
  //   S0D0F0  | 0X0 | 001 | 011 | 1X0 |
  // S1D1F1
  //  0X0      | 0X0 | 0X0 | 0X0 | 0X0 |
  //  001      | 0X0 | 001 | 001 | 001 |
  //  011      | 0X0 | 001 | 011 | 011 |
  //  1X0      | 0X0 | 001 | 011 | 1X0 |

  return make_mixed_bits(
      constant<uint32_t, S0 & S1>{},
      //(S0 | m0.dynamic_int_) & (S1 | m1.dynamic_int_),
      ((S1 & F0) & m0.dynamic_int_) | ((S0 & F1) & m1.dynamic_int_) |
          (m0.dynamic_int_ & m1.dynamic_int_),
      constant<uint32_t, (S1 & F0) | (S0 & F1) | (F0 & F1)>{});
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator&(MixedBits<S0, D0, F0> const& m,
                                          constant<TS1, S1> const&) {
  return make_mixed_bits(constant<uint32_t, S0 & S1>{},
                         m.dynamic_int_,
                         constant<uint32_t, S1 & F0>{});
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator&(constant<TS1, S1> const& s,
                                          MixedBits<S0, D0, F0> const& m) {
  return m & s;
}

// Bitwise OR
template <uint32_t S0,
          class D0,
          uint32_t F0,
          uint32_t S1,
          class D1,
          uint32_t F1>
CUTE_HOST_DEVICE constexpr auto operator|(MixedBits<S0, D0, F0> const& m0,
                                          MixedBits<S1, D1, F1> const& m1) {
  // Truth table for (S0,D0,F0) | (S1,D1,F1) -> (S,D,F)
  //   S0D0F0 | 0X0 | 001 | 011 | 1X0 |
  // S1D1F1
  //  0X0     | 0X0 | 001 | 011 | 1X0 |
  //  001     | 001 | 001 | 011 | 1X0 |
  //  011     | 011 | 011 | 011 | 1X0 |
  //  1X0     | 1X0 | 1X0 | 1X0 | 1X0 |

  return make_mixed_bits(
      constant<uint32_t, S0 | S1>{},
      ((~S1 & F0) & m0.dynamic_int_) | ((~S0 & F1) & m1.dynamic_int_),
      constant<uint32_t, (~S0 & F1) | (~S1 & F0)>{});
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator|(MixedBits<S0, D0, F0> const& m,
                                          constant<TS1, S1> const&) {
  return make_mixed_bits(constant<uint32_t, S0 | S1>{},
                         m.dynamic_int_,
                         constant<uint32_t, ~S1 & F0>{});
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator|(constant<TS1, S1> const& s,
                                          MixedBits<S0, D0, F0> const& m) {
  return m | s;
}

// Bitwise XOR
template <uint32_t S0,
          class D0,
          uint32_t F0,
          uint32_t S1,
          class D1,
          uint32_t F1>
CUTE_HOST_DEVICE constexpr auto operator^(MixedBits<S0, D0, F0> const& m0,
                                          MixedBits<S1, D1, F1> const& m1) {
  // Truth table for (S0,D0,F0) ^ (S1,D1,F1) -> (S,D,F)
  //   S0D0F0 | 0X0 | 001 | 011 | 1X0 |
  // S1D1F1
  //  0X0     | 0X0 | 001 | 011 | 1X0 |
  //  001     | 001 | 001 | 011 | 011 |
  //  011     | 011 | 011 | 001 | 001 |
  //  1X0     | 1X0 | 011 | 001 | 0X0 |

  return make_mixed_bits(
      constant<uint32_t, (~S0 & S1 & ~F0) | (S0 & ~S1 & ~F1)>{},
      (S0 | m0.dynamic_int_) ^ (S1 | m1.dynamic_int_),
      constant<uint32_t, F0 | F1>{});
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator^(MixedBits<S0, D0, F0> const& m,
                                          constant<TS1, S1> const&) {
  return make_mixed_bits(constant<uint32_t, (~S0 & S1 & ~F0) | (S0 & ~S1)>{},
                         (S0 | m.dynamic_int_) ^ S1,
                         constant<uint32_t, F0>{});
}

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto operator^(constant<TS1, S1> const& s,
                                          MixedBits<S0, D0, F0> const& m) {
  return m ^ s;
}

//
// upcast and downcast
//

template <uint32_t S0, class D0, uint32_t F0, class TS1, TS1 S1>
CUTE_HOST_DEVICE constexpr auto safe_div(MixedBits<S0, D0, F0> const& m,
                                         constant<TS1, S1> const& s) {
  static_assert(has_single_bit(S1), "Only divide MixedBits by powers of two.");
  return make_mixed_bits(safe_div(constant<uint32_t, S0>{}, s),
                         safe_div(m.dynamic_int_, s),
                         safe_div(constant<uint32_t, F0>{}, s));
}

template <uint32_t N, uint32_t S0, class D0, uint32_t F0>
CUTE_HOST_DEVICE constexpr auto upcast(MixedBits<S0, D0, F0> const& m) {
  static_assert(has_single_bit(N), "Only divide MixedBits by powers of two.");
  return safe_div(m, constant<uint32_t, N>{});
}

template <uint32_t N, class T, __CUTE_REQUIRES(cute::is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr auto upcast(T const& m) {
  return safe_div(m, constant<uint32_t, N>{});
}

template <uint32_t N, uint32_t S0, class D0, uint32_t F0>
CUTE_HOST_DEVICE constexpr auto downcast(MixedBits<S0, D0, F0> const& m) {
  static_assert(has_single_bit(N), "Only scale MixedBits by powers of two.");
  return make_mixed_bits(constant<uint32_t, S0 * N>{},
                         m.dynamic_int_ * N,
                         constant<uint32_t, F0 * N>{});
}

template <uint32_t N, class T, __CUTE_REQUIRES(cute::is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr auto downcast(T const& m) {
  return m * constant<uint32_t, N>{};
}

//
// Convert a Pow2Layout+Coord to a MixedBits
//

template <class Shape, class Stride, class Coord>
CUTE_HOST_DEVICE constexpr auto to_mixed_bits(Shape const& shape,
                                              Stride const& stride,
                                              Coord const& coord) {
  if constexpr (is_tuple<Shape>::value && is_tuple<Stride>::value &&
                is_tuple<Coord>::value) {
    static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value,
                  "Mismatched ranks");
    static_assert(tuple_size<Shape>::value == tuple_size<Coord>::value,
                  "Mismatched ranks");
    return transform_apply(
        shape,
        stride,
        coord,
        [](auto const& s, auto const& d, auto const& c) {
          return to_mixed_bits(s, d, c);
        },
        [](auto const&... a) { return (a ^ ...); });
  } else if constexpr (is_integral<Shape>::value &&
                       is_integral<Stride>::value &&
                       is_integral<Coord>::value) {
    static_assert(decltype(shape * stride)::value == 0 ||
                      has_single_bit(decltype(shape * stride)::value),
                  "Requires pow2 shape*stride.");
    return make_mixed_bits(
        Int<0>{}, coord * stride, (shape - Int<1>{}) * stride);
  } else {
    static_assert(is_integral<Shape>::value && is_integral<Stride>::value &&
                      is_integral<Coord>::value,
                  "Either Shape, Stride, and Coord must be all tuples, or they "
                  "must be all integral (in the sense of cute::is_integral).");
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Layout, class Coord>
CUTE_HOST_DEVICE constexpr auto to_mixed_bits(Layout const& layout,
                                              Coord const& coord) {
  return to_mixed_bits(
      layout.shape(), layout.stride(), idx2crd(coord, layout.shape()));
}

//
// Display utilities
//

template <uint32_t S, class D, uint32_t F>
CUTE_HOST_DEVICE void print(MixedBits<S, D, F> const& m) {
  printf("M_%u|(%u&%u)=%u", S, uint32_t(m.dynamic_int_), F, to_integral(m));
}

template <uint32_t S, class D, uint32_t F>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   MixedBits<S, D, F> const& m) {
  return os << "M_" << S << "|(" << uint32_t(m.dynamic_int_) << "&" << F
            << ")=" << to_integral(m);
}

template <int B, int M, int S>
CUTE_HOST_DEVICE void print(Swizzle<B, M, S> const&) {
  print("S<%d,%d,%d>", B, M, S);
}

template <int B, int M, int S>
CUTE_HOST std::ostream& operator<<(std::ostream& os, Swizzle<B, M, S> const&) {
  return os << "S<" << B << "," << M << "," << S << ">";
}

}  // end namespace cute
