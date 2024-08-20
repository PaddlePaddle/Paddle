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

//
// CUDA compatible print and printf
//

namespace cute {

CUTE_HOST_DEVICE
int num_digits(int x) {
  return (
      x < 10
          ? 1
          : (x < 100
                 ? 2
                 : (x < 1000
                        ? 3
                        : (x < 10000
                               ? 4
                               : (x < 100000
                                      ? 5
                                      : (x < 1000000
                                             ? 6
                                             : (x < 10000000
                                                    ? 7
                                                    : (x < 100000000
                                                           ? 8
                                                           : (x < 1000000000
                                                                  ? 9
                                                                  : 10)))))))));
}

template <class T>
struct format_and_size {
  using type = T;
  char const* format;
  int digits;
};

CUTE_HOST_DEVICE
format_and_size<int> get_format(bool) { return {"%*d", 3}; }

CUTE_HOST_DEVICE
format_and_size<int32_t> get_format(int32_t) { return {"%*d", 5}; }

CUTE_HOST_DEVICE
format_and_size<uint32_t> get_format(uint32_t) { return {"%*d", 5}; }

CUTE_HOST_DEVICE
format_and_size<int64_t> get_format(int64_t) { return {"%*d", 5}; }

CUTE_HOST_DEVICE
format_and_size<uint64_t> get_format(uint64_t) { return {"%*d", 5}; }

CUTE_HOST_DEVICE
format_and_size<float> get_format(half_t) { return {"%*.2f", 8}; }

CUTE_HOST_DEVICE
format_and_size<float> get_format(float) { return {"%*.2e", 10}; }

CUTE_HOST_DEVICE
format_and_size<double> get_format(double) { return {"%*.3e", 11}; }

//
// print dispatcher
//

CUTE_HOST_DEVICE
void print(char const& c) { printf("%c", c); }

template <class T, __CUTE_REQUIRES(std::is_integral<T>::value)>
CUTE_HOST_DEVICE void print(T const& a) {
  printf("%d", int(a));
}

template <class... T>
CUTE_HOST_DEVICE void print(char const* format, T const&... t) {
  printf(format, t...);
}

}  // end namespace cute
