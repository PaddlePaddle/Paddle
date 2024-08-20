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

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/container/tuple.hpp>
#include <cute/numeric/integral_constant.hpp>

namespace cute {

template <class... T>
struct ArithmeticTuple : tuple<T...> {
  template <class... U>
  CUTE_HOST_DEVICE constexpr ArithmeticTuple(ArithmeticTuple<U...> const& u)
      : tuple<T...>(static_cast<tuple<U...> const&>(u)) {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr ArithmeticTuple(tuple<U...> const& u)
      : tuple<T...>(u) {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr ArithmeticTuple(U const&... u)
      : tuple<T...>(u...) {}
};

template <class... T>
struct is_tuple<ArithmeticTuple<T...>> : true_type {};

template <class... T>
CUTE_HOST_DEVICE constexpr auto make_arithmetic_tuple(T const&... t) {
  return ArithmeticTuple<T...>(t...);
}

template <class... T>
CUTE_HOST_DEVICE constexpr auto as_arithmetic_tuple(tuple<T...> const& t) {
  return ArithmeticTuple<T...>(t);
}

//
// Numeric operators
//

// Addition
template <class... T, class... U>
CUTE_HOST_DEVICE constexpr auto operator+(ArithmeticTuple<T...> const& t,
                                          ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(
      append<R>(t, Int<0>{}),
      append<R>(u, Int<0>{}),
      plus{},
      [](auto const&... a) { return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr auto operator+(ArithmeticTuple<T...> const& t,
                                          tuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(
      append<R>(t, Int<0>{}),
      append<R>(u, Int<0>{}),
      plus{},
      [](auto const&... a) { return make_arithmetic_tuple(a...); });
}

template <class... T, class... U>
CUTE_HOST_DEVICE constexpr auto operator+(tuple<T...> const& t,
                                          ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), int(sizeof...(U)));
  return transform_apply(
      append<R>(t, Int<0>{}),
      append<R>(u, Int<0>{}),
      plus{},
      [](auto const&... a) { return make_arithmetic_tuple(a...); });
}

//
// Special cases
//

template <class T, class... U>
CUTE_HOST_DEVICE constexpr auto operator+(constant<T, 0>,
                                          ArithmeticTuple<U...> const& u) {
  return u;
}

template <class... T, class U>
CUTE_HOST_DEVICE constexpr auto operator+(ArithmeticTuple<T...> const& t,
                                          constant<U, 0>) {
  return t;
}

//
// ArithmeticTupleIterator
//

template <class ArithTuple>
struct ArithmeticTupleIterator {
  ArithTuple coord_;

  CUTE_HOST_DEVICE constexpr ArithmeticTupleIterator() : coord_() {}
  CUTE_HOST_DEVICE constexpr ArithmeticTupleIterator(ArithTuple const& coord)
      : coord_(coord) {}

  CUTE_HOST_DEVICE constexpr ArithTuple const& operator*() const {
    return coord_;
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr auto operator+(Coord const& c) const {
    return ArithmeticTupleIterator<decltype(coord_ + c)>(coord_ + c);
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr auto operator[](Coord const& c) const {
    return *(*this + c);
  }
};

template <class ArithTuple>
CUTE_HOST_DEVICE void print(ArithmeticTupleIterator<ArithTuple> const& iter) {
  printf("ArithTuple");
  print(iter.coord_);
}

//
// ArithmeticTuple "basis" elements
//

// Abstract value:
//   A ScaledBasis<T,N> is a (at least) rank-N0 ArithmeticTuple:
//      (_0,_0,...,T,_0,...)

template <class T, int N>
struct ScaledBasis : private tuple<T> {
  CUTE_HOST_DEVICE constexpr ScaledBasis(T const& t = {}) : tuple<T>(t) {}

  CUTE_HOST_DEVICE constexpr decltype(auto) value() {
    return get<0>(static_cast<tuple<T>&>(*this));
  }
  CUTE_HOST_DEVICE constexpr decltype(auto) value() const {
    return get<0>(static_cast<tuple<T> const&>(*this));
  }

  CUTE_HOST_DEVICE static constexpr auto mode() { return Int<N>{}; }
};

template <class T>
struct is_scaled_basis : false_type {};
template <class T, int N>
struct is_scaled_basis<ScaledBasis<T, N>> : true_type {};

template <class T, int N>
struct is_integral<ScaledBasis<T, N>> : true_type {};

template <class T>
CUTE_HOST_DEVICE constexpr auto basis_value(T const& e) {
  return e;
}

template <class T, int N>
CUTE_HOST_DEVICE constexpr auto basis_value(ScaledBasis<T, N> const& e) {
  return basis_value(e.value());
}

namespace detail {

template <int... Ns>
struct Basis;

template <>
struct Basis<> {
  using type = Int<1>;
};

template <int N, int... Ns>
struct Basis<N, Ns...> {
  using type = ScaledBasis<typename Basis<Ns...>::type, N>;
};

}  // end namespace detail

template <int... N>
using E = typename detail::Basis<N...>::type;

namespace detail {

template <class T, int... I, int... J>
CUTE_HOST_DEVICE constexpr auto as_arithmetic_tuple(T const& t,
                                                    seq<I...>,
                                                    seq<J...>) {
  return make_arithmetic_tuple(
      (void(I), Int<0>{})..., t, (void(J), Int<0>{})...);
}

template <class... T, int... I, int... J>
CUTE_HOST_DEVICE constexpr auto as_arithmetic_tuple(
    ArithmeticTuple<T...> const& t, seq<I...>, seq<J...>) {
  return make_arithmetic_tuple(get<I>(t)..., (void(J), Int<0>{})...);
}

}  // end namespace detail

// Turn a ScaledBases<T,N> into a rank-M ArithmeticTuple
//    with N prefix 0s:  (_0,_0,...N...,_0,T,_0,...,_0,_0)
template <int M, class T, int N>
CUTE_HOST_DEVICE constexpr auto as_arithmetic_tuple(
    ScaledBasis<T, N> const& t) {
  static_assert(M > N, "Mismatched ranks");
  return detail::as_arithmetic_tuple(
      t.value(), make_seq<N>{}, make_seq<M - N - 1>{});
}

// Turn an ArithmeticTuple into a rank-M ArithmeticTuple
//    with postfix 0s:  (t0,t1,t2,...,_0,...,_0,_0)
template <int M, class... T>
CUTE_HOST_DEVICE constexpr auto as_arithmetic_tuple(
    ArithmeticTuple<T...> const& t) {
  static_assert(M >= sizeof...(T), "Mismatched ranks");
  return detail::as_arithmetic_tuple(
      t, make_seq<int(sizeof...(T))>{}, make_seq<M - int(sizeof...(T))>{});
}

// Return...
template <class Shape>
CUTE_HOST_DEVICE constexpr auto make_basis_like(Shape const& shape) {
  if constexpr (is_integral<Shape>::value) {
    return Int<1>{};
  } else {
    // Generate bases for each rank of shape
    return transform(tuple_seq<Shape>{}, [&](auto I) {
      // Generate bases for each rank of shape_i and add an i on front
      constexpr int i = decltype(I)::value;  // NOTE: nvcc workaround
      return transform_leaf(make_basis_like(get<i>(shape)), [&](auto e) {
        return ScaledBasis<decltype(e), i>{};
      });
    });
  }

  CUTE_GCC_UNREACHABLE;
}

// Equality
template <class T, int N, int M>
CUTE_HOST_DEVICE constexpr auto operator==(ScaledBasis<T, N>, Int<M>) {
  return false_type{};
}

template <int N, class U, int M>
CUTE_HOST_DEVICE constexpr auto operator==(Int<N>, ScaledBasis<U, M>) {
  return false_type{};
}

template <class T, int N, class U, int M>
CUTE_HOST_DEVICE constexpr auto operator==(ScaledBasis<T, N> const& t,
                                           ScaledBasis<U, M> const& u) {
  return bool_constant<M == N>{} && t.value() == u.value();
}

// Multiplication
template <class A, int N, class T, __CUTE_REQUIRES(cute::is_integral<A>::value)>
CUTE_HOST_DEVICE constexpr auto operator*(A const& a,
                                          ScaledBasis<T, N> const& e) {
  return ScaledBasis<decltype(a * e.value()), N>{a * e.value()};
}

template <int N, class T, class B, __CUTE_REQUIRES(cute::is_integral<B>::value)>
CUTE_HOST_DEVICE constexpr auto operator*(ScaledBasis<T, N> const& e,
                                          B const& b) {
  return ScaledBasis<decltype(e.value() * b), N>{e.value() * b};
}

// Addition
template <int N, class T, class... U>
CUTE_HOST_DEVICE constexpr auto operator+(ScaledBasis<T, N> const& t,
                                          ArithmeticTuple<U...> const& u) {
  constexpr int R = cute::max(N + 1, int(sizeof...(U)));
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple<R>(u);
}

template <class... T, int M, class U>
CUTE_HOST_DEVICE constexpr auto operator+(ArithmeticTuple<T...> const& t,
                                          ScaledBasis<U, M> const& u) {
  constexpr int R = cute::max(int(sizeof...(T)), M + 1);
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple<R>(u);
}

template <int N, class T, int M, class U>
CUTE_HOST_DEVICE constexpr auto operator+(ScaledBasis<T, N> const& t,
                                          ScaledBasis<U, M> const& u) {
  constexpr int R = cute::max(N + 1, M + 1);
  return as_arithmetic_tuple<R>(t) + as_arithmetic_tuple<R>(u);
}

template <class T, class U, int M>
CUTE_HOST_DEVICE constexpr auto operator+(constant<T, 0>,
                                          ScaledBasis<U, M> const& u) {
  return u;
}

template <class T, int N, class U>
CUTE_HOST_DEVICE constexpr auto operator+(ScaledBasis<T, N> const& t,
                                          constant<U, 0>) {
  return t;
}

//
// Display utilities
//

template <class T, int N>
CUTE_HOST_DEVICE void print(ScaledBasis<T, N> const& e) {
  printf("%d:", N);
  print(e.value());
}

template <class T, int N>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   ScaledBasis<T, N> const& e) {
  return os << N << ":" << e.value();
}

}  // end namespace cute

namespace std {

template <class... T>
struct tuple_size<cute::ArithmeticTuple<T...>>
    : std::integral_constant<std::size_t, sizeof...(T)> {};

template <std::size_t I, class... T>
struct tuple_element<I, cute::ArithmeticTuple<T...>>
    : std::tuple_element<I, std::tuple<T...>> {};

}  // end namespace std
