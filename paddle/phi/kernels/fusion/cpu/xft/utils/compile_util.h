// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <type_traits>
#include <iostream>
#include <cstdio>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define REQUIRES(assertion, format, ...)            \
    do {                                            \
        if (unlikely(!(assertion))) {               \
            fprintf(stderr, format, ##__VA_ARGS__); \
            fprintf(stderr, "\n");                  \
            exit(-1);                               \
        }                                           \
    } while (0)

// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda &function, Args... args) {
        compile_time_for<i - 1>::op(function, args...);
        function(std::integral_constant<int, i - 1> {}, args...);
    }
};
template <>
struct compile_time_for<1> {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda &function, Args... args) {
        function(std::integral_constant<int, 0> {}, args...);
    }
};
template <>
struct compile_time_for<0> {
    // 0 loops, do nothing
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda &function, Args... args) {}
};