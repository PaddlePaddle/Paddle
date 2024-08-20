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

#include <tuple>
#include <utility>

#include <cute/config.hpp>
#include <cute/util/type_traits.hpp>

#include <cute/numeric/integral_constant.hpp>  // cute::true_type, cute::false_type
//#include <cute/container/array.hpp>            // Advanced optimizations

#if 0
//
// Use of agency::tuple is functional, but is over-engineered for our purposes...
//   This tends to result in slow compilation times and unintentionally propagated cvref types
//

#include <agency/tuple.hpp>

namespace cute
{

using agency::tuple;

using agency::make_tuple;
using agency::tuple_cat;

} // end namespace cute
#endif

// cute::tuple is like std::tuple, with two differences.
//
// 1. It works on both host and device.
// 2. Its template arguments must be semiregular types.
//
// Semiregular types are default constructible and copyable.
// They include "value types" like int or float,
// but do _not_ include references like int& or float&.
// (See std::tie for an example of a tuple of references.)
//
// This is simplified over the implementation in std:: and agency:: by ignoring
// much of
//    the conversion SFINAE, special overloading, and avoiding cvref template
//    types. Furthermore, the empty base optimization (EBO) is MORE aggressive
//    by avoiding construction calls, and ignoring any need for unique element
//    addresses.
//
// Over the agency::tuple implementation, this appears to accelerate compilation
// times by over 3x.

namespace cute {

namespace detail {

// EBO stands for "empty base optimization."
// We use this technique to ensure that cute::tuple
// doesn't need to waste space storing any template arguments
// of cute::tuple that have no data (like integral_constant).
// Otherwise, cute::tuple would need to spend at least 1 byte
// for each of its template arguments.
//
// EBO always "holds" a single value of type T.
// N is like an array index that TupleBase uses
// to access the desired tuple element.
template <std::size_t N, class T, bool IsEmpty = std::is_empty<T>::value>
struct EBO;

// Specialization for types T that have no data;
// the "static tuple leaf."  Valid T here include
// integral_constant<U, Value>, Int<Value>,
// and any other semiregular type
// for which std::is_empty_v<T> is true.
template <std::size_t N, class T>
struct EBO<N, T, true> {
  CUTE_HOST_DEVICE constexpr EBO() {}

  CUTE_HOST_DEVICE constexpr EBO(T const&) {}
};

template <std::size_t N, class T>
CUTE_HOST_DEVICE constexpr T getv(EBO<N, T, true> const&) {
  return {};
}

// Specialization for types T that are not empty;
// the "dynamic tuple leaf."  Valid T here include int,
// any other integral or floating-point type,
// or any semiregular type for which std::is_empty_v<T> is false.
template <std::size_t N, class T>
struct EBO<N, T, false> {
  CUTE_HOST_DEVICE constexpr EBO() : t_{} {}

  template <class U>
  CUTE_HOST_DEVICE constexpr EBO(U const& u) : t_{u} {}

  T t_;
};

template <std::size_t N, class T>
CUTE_HOST_DEVICE constexpr T const& getv(EBO<N, T, false> const& x) {
  return x.t_;
}

template <std::size_t N, class T>
CUTE_HOST_DEVICE constexpr T& getv(EBO<N, T, false>& x) {
  return x.t_;
}

template <std::size_t N, class T>
CUTE_HOST_DEVICE constexpr T&& getv(EBO<N, T, false>&& x) {
  return static_cast<T&&>(x.t_);
}

template <class IdxSeq, class... T>
struct TupleBase;

// Base class of cute::tuple.
// It inherits from EBO<i, t> for each (i, t) in (I..., T...).
// The actual storage (for nonempty t) lives in the base classes.
// index_sequence is a way to wrap up a sequence of zero or more
// compile-time integer values in a single type.
// We only ever use index_sequence<0, 1, ..., sizeof...(T)> in practice,
// as the type alias TupleBase below indicates.
template <std::size_t... I, class... T>
struct TupleBase<std::index_sequence<I...>, T...> : EBO<I, T>... {
  CUTE_HOST_DEVICE constexpr TupleBase() {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr explicit TupleBase(U const&... u)
      : EBO<I, T>(u)... {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr TupleBase(
      TupleBase<std::index_sequence<I...>, U...> const& u)
      : EBO<I, T>(getv(static_cast<EBO<I, U> const&>(u)))... {}
};

}  // end namespace detail

// make_index_sequence<K> returns index_sequence<0, 1, ..., K-1>.
template <class... T>
using TupleBase =
    detail::TupleBase<std::make_index_sequence<sizeof...(T)>, T...>;

// This is the actual cute::tuple class.
// The storage (if any) lives in TupleBase's EBO base classes.
template <class... T>
struct tuple : TupleBase<T...> {
  CUTE_HOST_DEVICE constexpr tuple() {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr tuple(U const&... u) : TupleBase<T...>(u...) {}

  template <class... U>
  CUTE_HOST_DEVICE constexpr tuple(tuple<U...> const& u)
      : TupleBase<T...>(static_cast<TupleBase<U...> const&>(u)) {}
};

//
// get for cute::tuple (just like std::get for std::tuple)
//

template <std::size_t I, class... T>
CUTE_HOST_DEVICE constexpr decltype(auto) get(tuple<T...> const& t) noexcept {
  static_assert(I < sizeof...(T), "Index out of range");
  return detail::getv<I>(t);
}

template <std::size_t I, class... T>
CUTE_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>& t) noexcept {
  static_assert(I < sizeof...(T), "Index out of range");
  return detail::getv<I>(t);
}

template <std::size_t I, class... T>
CUTE_HOST_DEVICE constexpr decltype(auto) get(tuple<T...>&& t) noexcept {
  static_assert(I < sizeof...(T), "Index out of range");
  return detail::getv<I>(static_cast<tuple<T...>&&>(t));
}

//
// Custom is_tuple trait simply checks the existence of std::tuple_size
//      and assumes std::get<I>(.), std::tuple_element<I,.>
//
namespace detail {

template <class T>
std::integral_constant<bool, std::tuple_size<T>::value >= 0> has_tuple_size(
    int);

template <class T>
std::false_type has_tuple_size(...);

}  // end namespace detail

template <class T>
struct is_tuple : decltype(detail::has_tuple_size<T>(0)) {};

//
// make_tuple (value-based implementation)
//

template <class... T>
CUTE_HOST_DEVICE constexpr tuple<T...> make_tuple(T const&... t) {
  return {t...};
}

//
// tuple_cat concatenates multiple cute::tuple into a single cute::tuple,
// just like std::tuple_cat for std::tuple.
//

#if 0
// Original implementation

namespace detail {

template <class T0, class T1,
          std::size_t... I0, std::size_t... I1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1,
          std::index_sequence<I0...>, std::index_sequence<I1...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

} // end namespace detail

CUTE_HOST_DEVICE constexpr
tuple<>
tuple_cat()
{
  return {};
}

template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE constexpr
Tuple const&
tuple_cat(Tuple const& t)
{
  return t;
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1)
{
  return detail::tuple_cat(t0, t1,
                           std::make_index_sequence<std::tuple_size<T0>::value>{},
                           std::make_index_sequence<std::tuple_size<T1>::value>{});
}

template <class T0, class T1, class T2, class... Ts>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1, T2 const& t2, Ts const&... ts)
{
  return cute::tuple_cat(cute::tuple_cat(t0,t1),t2,ts...);
}
#endif

#if 1
// Extended implementation

namespace detail {

template <class T0, class T1, std::size_t... I0, std::size_t... I1>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>) {
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

template <class T0,
          class T1,
          class T2,
          std::size_t... I0,
          std::size_t... I1,
          std::size_t... I2>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          T2 const& t2,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>,
                                          std::index_sequence<I2...>) {
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)...);
}

template <class T0,
          class T1,
          class T2,
          class T3,
          std::size_t... I0,
          std::size_t... I1,
          std::size_t... I2,
          std::size_t... I3>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          T2 const& t2,
                                          T3 const& t3,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>,
                                          std::index_sequence<I2...>,
                                          std::index_sequence<I3...>) {
  return cute::make_tuple(
      get<I0>(t0)..., get<I1>(t1)..., get<I2>(t2)..., get<I3>(t3)...);
}

template <class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          std::size_t... I0,
          std::size_t... I1,
          std::size_t... I2,
          std::size_t... I3,
          std::size_t... I4>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          T2 const& t2,
                                          T3 const& t3,
                                          T4 const& t4,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>,
                                          std::index_sequence<I2...>,
                                          std::index_sequence<I3...>,
                                          std::index_sequence<I4...>) {
  return cute::make_tuple(get<I0>(t0)...,
                          get<I1>(t1)...,
                          get<I2>(t2)...,
                          get<I3>(t3)...,
                          get<I4>(t4)...);
}

}  // end namespace detail

CUTE_HOST_DEVICE constexpr tuple<> tuple_cat() { return {}; }

template <class Tuple, __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE constexpr Tuple const& tuple_cat(Tuple const& t) {
  return t;
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0, T1 const& t1) {
  return detail::tuple_cat(
      t0,
      t1,
      std::make_index_sequence<std::tuple_size<T0>::value>{},
      std::make_index_sequence<std::tuple_size<T1>::value>{});
}

template <class T0, class T1, class T2>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          T2 const& t2) {
  return detail::tuple_cat(
      t0,
      t1,
      t2,
      std::make_index_sequence<std::tuple_size<T0>::value>{},
      std::make_index_sequence<std::tuple_size<T1>::value>{},
      std::make_index_sequence<std::tuple_size<T2>::value>{});
}

template <class T0, class T1, class T2, class T3>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          T2 const& t2,
                                          T3 const& t3) {
  return detail::tuple_cat(
      t0,
      t1,
      t2,
      t3,
      std::make_index_sequence<std::tuple_size<T0>::value>{},
      std::make_index_sequence<std::tuple_size<T1>::value>{},
      std::make_index_sequence<std::tuple_size<T2>::value>{},
      std::make_index_sequence<std::tuple_size<T3>::value>{});
}

template <class T0, class T1, class T2, class T3, class T4>
CUTE_HOST_DEVICE constexpr auto tuple_cat(
    T0 const& t0, T1 const& t1, T2 const& t2, T3 const& t3, T4 const& t4) {
  return detail::tuple_cat(
      t0,
      t1,
      t2,
      t3,
      t4,
      std::make_index_sequence<std::tuple_size<T0>::value>{},
      std::make_index_sequence<std::tuple_size<T1>::value>{},
      std::make_index_sequence<std::tuple_size<T2>::value>{},
      std::make_index_sequence<std::tuple_size<T3>::value>{},
      std::make_index_sequence<std::tuple_size<T4>::value>{});
}

template <class T0,
          class T1,
          class T2,
          class T3,
          class T4,
          class T5,
          class... Ts>
CUTE_HOST_DEVICE constexpr auto tuple_cat(T0 const& t0,
                                          T1 const& t1,
                                          T2 const& t2,
                                          T3 const& t3,
                                          T4 const& t4,
                                          T5 const& t5,
                                          Ts const&... ts) {
  return cute::tuple_cat(cute::tuple_cat(t0, t1, t2, t3, t4), t5, ts...);
}
#endif

#if 0
// Outer-Inner indexing trick to concat all tuples at once

namespace detail {

template <std::size_t... Ns>
struct tuple_cat_helper
{
  static constexpr cute::array<std::size_t,sizeof...(Ns)> ns = {Ns...};

  static constexpr std::size_t total_size() {
    std::size_t sum = 0;
    for (std::size_t n : ns) sum += n;
    return sum;
  }
  static constexpr std::size_t total_size_ = total_size();

  static constexpr auto values() {
    cute::array<std::size_t[2],total_size_> outer_inner = {};

    std::size_t idx = 0;
    for (std::size_t i = 0; i < ns.size(); ++i) {
      for (std::size_t j = 0; j < ns[i]; ++j, ++idx) {
        outer_inner[idx][0] = i;
        outer_inner[idx][1] = j;
      }
    }
    return outer_inner;
  }
  static constexpr auto outer_inner_ = values();

  using total_sequence = std::make_index_sequence<total_size_>;
};

template <class Helper, class Tuple, std::size_t... I>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(Tuple const& t, std::index_sequence<I...>)
{
  return cute::make_tuple(get<Helper::outer_inner_[I][1]>(get<Helper::outer_inner_[I][0]>(t))...);
}

template <class T0, class T1,
          std::size_t... I0, std::size_t... I1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1,
          std::index_sequence<I0...>, std::index_sequence<I1...>)
{
  return cute::make_tuple(get<I0>(t0)..., get<I1>(t1)...);
}

} // end namespace detail

CUTE_HOST_DEVICE constexpr
tuple<>
tuple_cat()
{
  return {};
}

template <class Tuple,
          __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE constexpr
Tuple const&
tuple_cat(Tuple const& t)
{
  return t;
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(T0 const& t0, T1 const& t1)
{
  return detail::tuple_cat(t0, t1,
                           std::make_index_sequence<std::tuple_size<T0>::value>{},
                           std::make_index_sequence<std::tuple_size<T1>::value>{});
}

template <class... Tuples>
CUTE_HOST_DEVICE constexpr
auto
tuple_cat(Tuples const&... ts)
{
  using Helper = detail::tuple_cat_helper<std::tuple_size<Tuples>::value...>;
  return detail::tuple_cat<Helper>(make_tuple(ts...), typename Helper::total_sequence{});
}
#endif

//
// Equality operators
//

namespace detail {

template <std::size_t I, class TupleA, class TupleB>
CUTE_HOST_DEVICE constexpr auto equal_impl(TupleA const& a, TupleB const& b) {
  if constexpr (I == std::tuple_size<TupleA>::value) {
    return cute::true_type{};  // Terminal: TupleA is exhausted
  } else if constexpr (I == std::tuple_size<TupleB>::value) {
    return cute::false_type{};  // Terminal: TupleA is not exhausted, TupleB is
                                // exhausted
  } else {
    return (get<I>(a) == get<I>(b)) && equal_impl<I + 1>(a, b);
  }

  CUTE_GCC_UNREACHABLE;
}

}  // end namespace detail

template <class TupleT,
          class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value&& is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr auto operator==(TupleT const& t, TupleU const& u) {
  return detail::equal_impl<0>(t, u);
}

template <class TupleT,
          class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value ^ is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr auto operator==(TupleT const& t, TupleU const& u) {
  return cute::false_type{};
}

template <class TupleT,
          class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value&& is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr auto operator!=(TupleT const& t, TupleU const& u) {
  return !(t == u);
}

template <class TupleT,
          class TupleU,
          __CUTE_REQUIRES(is_tuple<TupleT>::value ^ is_tuple<TupleU>::value)>
CUTE_HOST_DEVICE constexpr auto operator!=(TupleT const& t, TupleU const& u) {
  return cute::true_type{};
}

//
// Comparison operators
//

//
// There are many ways to compare tuple of elements and because CuTe is built
//   on parameterizing layouts of coordinates, some comparisons are appropriate
//   only in certain cases.
//  -- lexicographical comparison [reverse, reflected, revref]
//  -- colexicographical comparison [reverse, reflected, revref]
//  -- element-wise comparison [any,all]
// This can be very confusing. To avoid errors in selecting the appropriate
//   comparison, op<|op<=|op>|op>= are *not* implemented for cute::tuple.
//
// That said, see int_tuple for more explicitly named common comparison ops.
//

//
// Shortcuts
//

// using std::get;
using std::tuple_element;
using std::tuple_element_t;
using std::tuple_size;

//
// Display utilities
//

namespace detail {

template <class Tuple, std::size_t... Is>
CUTE_HOST_DEVICE void print_tuple(Tuple const& t,
                                  std::index_sequence<Is...>,
                                  char s = '(',
                                  char e = ')') {
  using eat = int[];
  using cute::print;
  (void)eat{(print(s), 0),
            (print(Is == 0 ? "" : ","), print(get<Is>(t)), 0)...,
            (print(e), 0)};
}

template <class Tuple, std::size_t... Is>
CUTE_HOST std::ostream& print_tuple_os(std::ostream& os,
                                       Tuple const& t,
                                       std::index_sequence<Is...>,
                                       char s = '(',
                                       char e = ')') {
  using eat = int[];
  (void)eat{(void(os << s), 0),
            (void(os << (Is == 0 ? "" : ",") << get<Is>(t)), 0)...,
            (void(os << e), 0)};
  return os;
}

}  // end namespace detail

template <class Tuple, __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST_DEVICE void print(Tuple const& t) {
  return detail::print_tuple(
      t, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

template <class Tuple, __CUTE_REQUIRES(is_tuple<Tuple>::value)>
CUTE_HOST std::ostream& operator<<(std::ostream& os, Tuple const& t) {
  return detail::print_tuple_os(
      os, t, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

}  // end namespace cute

//
// std:: compatability
//

namespace std {

template <class... T>
struct tuple_size<cute::tuple<T...>>
    : std::integral_constant<std::size_t, sizeof...(T)> {};

template <std::size_t I, class... T>
struct tuple_element<I, cute::tuple<T...>>
    : std::tuple_element<I, std::tuple<T...>> {};

}  // namespace std
