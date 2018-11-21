// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// UNUSED type.
// For unused local variable, you can just (void)(x);
#if defined(_MSC_VER)
#define UNUSED
#elif defined(__GNUC__) || defined(__clang__)
#define UNUSED __attribute__((unused))
#else
#error "Not supported compiler"
#endif

// Because most enforce conditions would evaluate to true, we can use
// __builtin_expect to instruct the C++ compiler to generate code that
// always forces branch prediction of true.
// This generates faster binary code. __builtin_expect is since C++11.
// For more details, please check https://stackoverflow.com/a/43870188/724872.
#if defined(__GNUC__) || defined(__clang__)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
#define UNLIKELY(condition) (condition)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#else
#define LIKELY(condition) (condition)
#endif
