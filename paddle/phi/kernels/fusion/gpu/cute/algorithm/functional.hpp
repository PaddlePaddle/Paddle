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

#include <utility>

#include <cute/config.hpp>

/** C++14 <functional> extensions */

namespace cute {

/**************/
/** Identity **/
/**************/

struct identity {
  template <class T>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const {
    return std::forward<T>(arg);
  }
};

template <class R>
struct constant_fn {
  template <class... T>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&&...) const {
    return r_;
  }
  R r_;
};

/***********/
/** Unary **/
/***********/

#define CUTE_LEFT_UNARY_OP(NAME, OP)                                      \
  struct NAME {                                                           \
    template <class T>                                                    \
    CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const { \
      return OP std::forward<T>(arg);                                     \
    }                                                                     \
  }
#define CUTE_RIGHT_UNARY_OP(NAME, OP)                                     \
  struct NAME {                                                           \
    template <class T>                                                    \
    CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const { \
      return std::forward<T>(arg) OP;                                     \
    }                                                                     \
  }
#define CUTE_NAMED_UNARY_OP(NAME, OP)                                     \
  struct NAME {                                                           \
    template <class T>                                                    \
    CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) const { \
      return OP(std::forward<T>(arg));                                    \
    }                                                                     \
  }

CUTE_LEFT_UNARY_OP(unary_plus, +);
CUTE_LEFT_UNARY_OP(negate, -);
CUTE_LEFT_UNARY_OP(bit_not, ~);
CUTE_LEFT_UNARY_OP(logical_not, !);
CUTE_LEFT_UNARY_OP(dereference, *);
CUTE_LEFT_UNARY_OP(address_of, &);
CUTE_LEFT_UNARY_OP(pre_increment, ++);
CUTE_LEFT_UNARY_OP(pre_decrement, --);

CUTE_RIGHT_UNARY_OP(post_increment, ++);
CUTE_RIGHT_UNARY_OP(post_decrement, --);

CUTE_NAMED_UNARY_OP(abs_fn, abs);
CUTE_NAMED_UNARY_OP(conjugate, cute::conj);

#undef CUTE_LEFT_UNARY_OP
#undef CUTE_RIGHT_UNARY_OP
#undef CUTE_NAMED_UNARY_OP

/************/
/** Binary **/
/************/

#define CUTE_BINARY_OP(NAME, OP)                                          \
  struct NAME {                                                           \
    template <class T, class U>                                           \
    CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& lhs,         \
                                                         U&& rhs) const { \
      return std::forward<T>(lhs) OP std::forward<U>(rhs);                \
    }                                                                     \
  }
#define CUTE_NAMED_BINARY_OP(NAME, OP)                                    \
  struct NAME {                                                           \
    template <class T, class U>                                           \
    CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& lhs,         \
                                                         U&& rhs) const { \
      return OP(std::forward<T>(lhs), std::forward<U>(rhs));              \
    }                                                                     \
  }

CUTE_BINARY_OP(plus, +);
CUTE_BINARY_OP(minus, -);
CUTE_BINARY_OP(multiplies, *);
CUTE_BINARY_OP(divides, /);
CUTE_BINARY_OP(modulus, %);

CUTE_BINARY_OP(plus_assign, +=);
CUTE_BINARY_OP(minus_assign, -=);
CUTE_BINARY_OP(multiplies_assign, *=);
CUTE_BINARY_OP(divides_assign, /=);
CUTE_BINARY_OP(modulus_assign, %=);

CUTE_BINARY_OP(bit_and, &);
CUTE_BINARY_OP(bit_or, |);
CUTE_BINARY_OP(bit_xor, ^);
CUTE_BINARY_OP(left_shift, <<);
CUTE_BINARY_OP(right_shift, >>);

CUTE_BINARY_OP(bit_and_assign, &=);
CUTE_BINARY_OP(bit_or_assign, |=);
CUTE_BINARY_OP(bit_xor_assign, ^=);
CUTE_BINARY_OP(left_shift_assign, <<=);
CUTE_BINARY_OP(right_shift_assign, >>=);

CUTE_BINARY_OP(logical_and, &&);
CUTE_BINARY_OP(logical_or, ||);

CUTE_BINARY_OP(equal_to, ==);
CUTE_BINARY_OP(not_equal_to, !=);
CUTE_BINARY_OP(greater, >);
CUTE_BINARY_OP(less, <);
CUTE_BINARY_OP(greater_equal, >=);
CUTE_BINARY_OP(less_equal, <=);

CUTE_NAMED_BINARY_OP(max_fn, cute::max);
CUTE_NAMED_BINARY_OP(min_fn, cute::min);

#undef CUTE_BINARY_OP
#undef CUTE_NAMED_BINARY_OP

/**********/
/** Meta **/
/**********/

template <class Fn, class Arg>
struct bound_fn {
  template <class T>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(T&& arg) {
    return fn_(arg_, std::forward<T>(arg));
  }

  Fn fn_;
  Arg arg_;
};

template <class Fn, class Arg>
CUTE_HOST_DEVICE constexpr auto bind(Fn const& fn, Arg const& arg) {
  return bound_fn<Fn, Arg>{fn, arg};
}

}  // end namespace cute
