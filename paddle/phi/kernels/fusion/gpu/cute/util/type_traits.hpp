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

#include <type_traits>

#include <cute/config.hpp>

#define __CUTE_REQUIRES(...) \
  typename std::enable_if<(__VA_ARGS__)>::type* = nullptr
#define __CUTE_REQUIRES_V(...) \
  typename std::enable_if<decltype((__VA_ARGS__))::value>::type* = nullptr

namespace cute {

using std::conjunction;
using std::conjunction_v;

using std::disjunction;
using std::disjunction_v;

using std::negation;
using std::negation_v;

using std::void_t;

// C++20
// using std::remove_cvref;
template <class T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

// C++20
// using std::remove_cvref_t;
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

//
// is_valid
//

namespace detail {

template <class F,
          class... Args,
          class = decltype(std::declval<F&&>()(std::declval<Args&&>()...))>
CUTE_HOST_DEVICE constexpr auto is_valid_impl(int) {
  return std::true_type{};
}

template <class F, class... Args>
CUTE_HOST_DEVICE constexpr auto is_valid_impl(...) {
  return std::false_type{};
}

template <class F>
struct is_valid_fn {
  template <class... Args>
  CUTE_HOST_DEVICE constexpr auto operator()(Args&&...) const {
    return is_valid_impl<F, Args&&...>(int{});
  }
};

}  // end namespace detail

template <class F>
CUTE_HOST_DEVICE constexpr auto is_valid(F&&) {
  return detail::is_valid_fn<F&&>{};
}

template <class F, class... Args>
CUTE_HOST_DEVICE constexpr auto is_valid(F&&, Args&&...) {
  return detail::is_valid_impl<F&&, Args&&...>(int{});
}

}  // end namespace cute
