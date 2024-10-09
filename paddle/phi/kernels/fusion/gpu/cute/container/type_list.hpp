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

namespace cute {

template <class T>
struct type_c {
  using type = T;
};

template <class... T>
struct type_list {};

}  // end namespace cute

//
// Specialize tuple-related functionality for cute::type_list
//

#include <cute/container/tuple.hpp>
#include <tuple>

namespace cute {

template <int I, class... T>
CUTE_HOST_DEVICE constexpr std::tuple_element_t<I, type_list<T...>> get(
    type_list<T...>&) noexcept {
  return {};
}
template <int I, class... T>
CUTE_HOST_DEVICE constexpr std::tuple_element_t<I, type_list<T...>> get(
    type_list<T...> const& t) noexcept {
  return {};
}

}  // end namespace cute

namespace std {

template <class... T>
struct tuple_size<cute::type_list<T...>>
    : std::integral_constant<std::size_t, sizeof...(T)> {};

template <std::size_t I, class... T>
struct tuple_element<I, cute::type_list<T...>>
    : cute::type_c<typename std::tuple_element<I, std::tuple<T...>>::type> {};

}  // end namespace std
