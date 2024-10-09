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

#include <utility>  // std::integer_sequence

#include <cute/config.hpp>

namespace cute {

using std::integer_sequence;
using std::make_integer_sequence;

namespace detail {

template <class T, class S, T Begin>
struct make_integer_range_impl;

template <class T, T... N, T Begin>
struct make_integer_range_impl<T, integer_sequence<T, N...>, Begin> {
  using type = integer_sequence<T, N + Begin...>;
};

}  // end namespace detail

template <class T, T Begin, T End>
using make_integer_range = typename detail::make_integer_range_impl<
    T,
    make_integer_sequence<T, (End - Begin > 0) ? (End - Begin) : 0>,
    Begin>::type;

//
// Common aliases
//

// int_sequence

template <int... Ints>
using int_sequence = integer_sequence<int, Ints...>;

template <int N>
using make_int_sequence = make_integer_sequence<int, N>;

template <int Begin, int End>
using make_int_range = make_integer_range<int, Begin, End>;

// index_sequence

template <std::size_t... Ints>
using index_sequence = integer_sequence<std::size_t, Ints...>;

template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

template <std::size_t Begin, std::size_t End>
using make_index_range = make_integer_range<std::size_t, Begin, End>;

//
// Shortcuts
//

template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

template <int Min, int Max>
using make_range = make_int_range<Min, Max>;

template <class Tuple>
using tuple_seq =
    make_seq<std::tuple_size<std::remove_reference_t<Tuple>>::value>;

}  // end namespace cute

//
// Specialize tuple-related functionality for cute::integer_sequence
//

#include <cute/numeric/integral_constant.hpp>
#include <tuple>

namespace cute {

template <std::size_t I, class T, T... Ints>
CUTE_HOST_DEVICE constexpr std::tuple_element_t<I, integer_sequence<T, Ints...>>
get(integer_sequence<T, Ints...>) {
  static_assert(I < sizeof...(Ints), "Index out of range");
  return {};
}

}  // end namespace cute

namespace std {

template <class T, T... Ints>
struct tuple_size<cute::integer_sequence<T, Ints...>>
    : std::integral_constant<std::size_t, sizeof...(Ints)> {};

template <std::size_t I, class T, T... Ints>
struct tuple_element<I, cute::integer_sequence<T, Ints...>>
    : std::tuple_element<I, std::tuple<cute::integral_constant<T, Ints>...>> {};

}  // end namespace std
