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

#include <cute/numeric/math.hpp>
#include <cute/util/type_traits.hpp>

namespace cute {

template <class T, T v>
struct constant : std::integral_constant<T, v> {
  static constexpr T value = v;
  using value_type = T;
  using type = constant<T, v>;
  CUTE_HOST_DEVICE constexpr operator value_type() const noexcept {
    return value;
  }
  CUTE_HOST_DEVICE constexpr value_type operator()() const noexcept {
    return value;
  }
};

template <class T, T v>
using integral_constant = constant<T, v>;

template <bool b>
using bool_constant = constant<bool, b>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

//
// Traits
//

// Use std::is_integral<T> to match built-in integral types (int, int64_t,
// unsigned, etc) Use cute::is_integral<T> to match both built-in integral types
// AND constant<T,t>

template <class T>
struct is_integral : bool_constant<std::is_integral<T>::value> {};
template <class T, T v>
struct is_integral<constant<T, v>> : true_type {};

// is_static detects if an (abstract) value is defined completely by it's type
// (no members)

template <class T>
struct is_static : bool_constant<std::is_empty<T>::value> {};

// is_constant detects if a type is a constant<T,v> and if v is equal to a value

template <auto n, class T>
struct is_constant : false_type {};
template <auto n, class T, T v>
struct is_constant<n, constant<T, v>> : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, constant<T, v> const> : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, constant<T, v> const&> : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, constant<T, v>&> : bool_constant<v == n> {};
template <auto n, class T, T v>
struct is_constant<n, constant<T, v>&&> : bool_constant<v == n> {};

//
// Specializations
//

template <int v>
using Int = constant<int, v>;

using _m32 = Int<-32>;
using _m24 = Int<-24>;
using _m16 = Int<-16>;
using _m12 = Int<-12>;
using _m10 = Int<-10>;
using _m9 = Int<-9>;
using _m8 = Int<-8>;
using _m7 = Int<-7>;
using _m6 = Int<-6>;
using _m5 = Int<-5>;
using _m4 = Int<-4>;
using _m3 = Int<-3>;
using _m2 = Int<-2>;
using _m1 = Int<-1>;
using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _3 = Int<3>;
using _4 = Int<4>;
using _5 = Int<5>;
using _6 = Int<6>;
using _7 = Int<7>;
using _8 = Int<8>;
using _9 = Int<9>;
using _10 = Int<10>;
using _12 = Int<12>;
using _16 = Int<16>;
using _24 = Int<24>;
using _32 = Int<32>;
using _64 = Int<64>;
using _96 = Int<96>;
using _128 = Int<128>;
using _192 = Int<192>;
using _256 = Int<256>;
using _512 = Int<512>;
using _1024 = Int<1024>;
using _2048 = Int<2048>;
using _4096 = Int<4096>;
using _8192 = Int<8192>;

/***************/
/** Operators **/
/***************/

#define CUTE_LEFT_UNARY_OP(OP)                                             \
  template <class T, T t>                                                  \
  CUTE_HOST_DEVICE constexpr constant<decltype(OP t), (OP t)> operator OP( \
      constant<T, t>) {                                                    \
    return {};                                                             \
  }
#define CUTE_RIGHT_UNARY_OP(OP)                                            \
  template <class T, T t>                                                  \
  CUTE_HOST_DEVICE constexpr constant<decltype(t OP), (t OP)> operator OP( \
      constant<T, t>) {                                                    \
    return {};                                                             \
  }

#define CUTE_BINARY_OP(OP)                                                     \
  template <class T, T t, class U, U u>                                        \
  CUTE_HOST_DEVICE constexpr constant<decltype(t OP u), (t OP u)> operator OP( \
      constant<T, t>, constant<U, u>) {                                        \
    return {};                                                                 \
  }

CUTE_LEFT_UNARY_OP(+);
CUTE_LEFT_UNARY_OP(-);
CUTE_LEFT_UNARY_OP(~);
CUTE_LEFT_UNARY_OP(!);
CUTE_LEFT_UNARY_OP(*);

CUTE_BINARY_OP(+);
CUTE_BINARY_OP(-);
CUTE_BINARY_OP(*);
CUTE_BINARY_OP(/);
CUTE_BINARY_OP(%);
CUTE_BINARY_OP(&);
CUTE_BINARY_OP(|);
CUTE_BINARY_OP(^);
CUTE_BINARY_OP(<<);
CUTE_BINARY_OP(>>);

CUTE_BINARY_OP(&&);
CUTE_BINARY_OP(||);

CUTE_BINARY_OP(==);
CUTE_BINARY_OP(!=);
CUTE_BINARY_OP(>);
CUTE_BINARY_OP(<);
CUTE_BINARY_OP(>=);
CUTE_BINARY_OP(<=);

#undef CUTE_BINARY_OP
#undef CUTE_LEFT_UNARY_OP
#undef CUTE_RIGHT_UNARY_OP

//
// Mixed static-dynamic special cases
//

template <class T, class U, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator*(constant<T, 0>, U) {
  return {};
}

template <class U, class T, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator*(U, constant<T, 0>) {
  return {};
}

template <class T, class U, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator/(constant<T, 0>, U) {
  return {};
}

template <class U, class T, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator%(U, constant<T, 1>) {
  return {};
}

template <class U, class T, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator%(U, constant<T, -1>) {
  return {};
}

template <class T, class U, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator%(constant<T, 0>, U) {
  return {};
}

template <class T, class U, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator&(constant<T, 0>, U) {
  return {};
}

template <class T, class U, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr constant<T, 0> operator&(U, constant<T, 0>) {
  return {};
}

template <class T,
          T t,
          class U,
          __CUTE_REQUIRES(std::is_integral<U>::value && !bool(t))>
CUTE_HOST_DEVICE constexpr constant<bool, false> operator&&(constant<T, t>, U) {
  return {};
}

template <class T,
          T t,
          class U,
          __CUTE_REQUIRES(std::is_integral<U>::value && !bool(t))>
CUTE_HOST_DEVICE constexpr constant<bool, false> operator&&(U, constant<T, t>) {
  return {};
}

template <class T,
          class U,
          T t,
          __CUTE_REQUIRES(std::is_integral<U>::value&& bool(t))>
CUTE_HOST_DEVICE constexpr constant<bool, true> operator||(constant<T, t>, U) {
  return {};
}

template <class T,
          class U,
          T t,
          __CUTE_REQUIRES(std::is_integral<U>::value&& bool(t))>
CUTE_HOST_DEVICE constexpr constant<bool, true> operator||(U, constant<T, t>) {
  return {};
}

//
// Named functions from math.hpp
//

#define CUTE_NAMED_UNARY_FN(OP)                                   \
  template <class T, T t>                                         \
  CUTE_HOST_DEVICE constexpr constant<decltype(OP(t)), OP(t)> OP( \
      constant<T, t>) {                                           \
    return {};                                                    \
  }

#define CUTE_NAMED_BINARY_FN(OP)                                        \
  template <class T, T t, class U, U u>                                 \
  CUTE_HOST_DEVICE constexpr constant<decltype(OP(t, u)), OP(t, u)> OP( \
      constant<T, t>, constant<U, u>) {                                 \
    return {};                                                          \
  }                                                                     \
                                                                        \
  template <class T,                                                    \
            T t,                                                        \
            class U,                                                    \
            __CUTE_REQUIRES(std::is_integral<U>::value)>                \
  CUTE_HOST_DEVICE constexpr auto OP(constant<T, t>, U u) {             \
    return OP(t, u);                                                    \
  }                                                                     \
                                                                        \
  template <class T,                                                    \
            class U,                                                    \
            U u,                                                        \
            __CUTE_REQUIRES(std::is_integral<T>::value)>                \
  CUTE_HOST_DEVICE constexpr auto OP(T t, constant<U, u>) {             \
    return OP(t, u);                                                    \
  }

CUTE_NAMED_UNARY_FN(abs);
CUTE_NAMED_UNARY_FN(signum);
CUTE_NAMED_UNARY_FN(has_single_bit);

CUTE_NAMED_BINARY_FN(max);
CUTE_NAMED_BINARY_FN(min);
CUTE_NAMED_BINARY_FN(shiftl);
CUTE_NAMED_BINARY_FN(shiftr);
CUTE_NAMED_BINARY_FN(gcd);
CUTE_NAMED_BINARY_FN(lcm);

#undef CUTE_NAMED_UNARY_FN
#undef CUTE_NAMED_BINARY_FN

//
// Other functions
//

template <class T, T t, class U, U u>
CUTE_HOST_DEVICE constexpr constant<decltype(t / u), t / u> safe_div(
    constant<T, t>, constant<U, u>) {
  static_assert(t % u == 0, "Static safe_div requires t % u == 0");
  return {};
}

template <class T, T t, class U, __CUTE_REQUIRES(std::is_integral<U>::value)>
CUTE_HOST_DEVICE constexpr auto safe_div(constant<T, t>, U u) {
  return t / u;
}

template <class T, class U, U u, __CUTE_REQUIRES(std::is_integral<T>::value)>
CUTE_HOST_DEVICE constexpr auto safe_div(T t, constant<U, u>) {
  return t / u;
}

// cute::true_type prefers standard conversion to std::true_type
//   over user-defined conversion to bool
template <class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr decltype(auto) conditional_return(std::true_type,
                                                             TrueType&& t,
                                                             FalseType&&) {
  return static_cast<TrueType&&>(t);
}

// cute::false_type prefers standard conversion to std::false_type
//   over user-defined conversion to bool
template <class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr decltype(auto) conditional_return(std::false_type,
                                                             TrueType&&,
                                                             FalseType&& f) {
  return static_cast<FalseType&&>(f);
}

// TrueType and FalseType must have a common type
template <class TrueType, class FalseType>
CUTE_HOST_DEVICE constexpr auto conditional_return(bool b,
                                                   TrueType const& t,
                                                   FalseType const& f) {
  return b ? t : f;
}

//
// Display utilities
//

template <class T, T N>
CUTE_HOST_DEVICE void print(integral_constant<T, N> const&) {
  printf("_%d", N);
}

template <class T, T N>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   integral_constant<T, N> const&) {
  return os << "_" << N;
}

}  // end namespace cute
