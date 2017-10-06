/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/utils/CpuId.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"

using namespace paddle;  // NOLINT

TEST(SIMDFlags, gccTest) {
#if (defined(__GNUC__) || defined(__GNUG__)) && !(defined(__clang__)) && \
    !defined(__arm__) && !defined(__aarch64__)
  // clang-format off
  CHECK(!__builtin_cpu_supports("sse")    != HAS_SSE);
  CHECK(!__builtin_cpu_supports("sse2")   != HAS_SSE2);
  CHECK(!__builtin_cpu_supports("sse3")   != HAS_SSE3);
  CHECK(!__builtin_cpu_supports("ssse3")  != HAS_SSSE3);
  CHECK(!__builtin_cpu_supports("sse4.1") != HAS_SSE41);
  CHECK(!__builtin_cpu_supports("sse4.2") != HAS_SSE42);
  CHECK(!__builtin_cpu_supports("avx")    != HAS_AVX);
  CHECK(!__builtin_cpu_supports("avx2")   != HAS_AVX2);
// clang-format on
#endif
}

TEST(SIMDFlags, normalPrint) {
  LOG(INFO) << "Has SSE:     " << std::boolalpha << HAS_SSE;
  LOG(INFO) << "Has SSE2:    " << std::boolalpha << HAS_SSE2;
  LOG(INFO) << "Has SSE3:    " << std::boolalpha << HAS_SSE3;
  LOG(INFO) << "Has SSSE3:   " << std::boolalpha << HAS_SSSE3;
  LOG(INFO) << "Has SSE4:    " << std::boolalpha << HAS_SSE41 || HAS_SSE42;
  LOG(INFO) << "Has FMA3:    " << std::boolalpha << HAS_FMA3;
  LOG(INFO) << "Has FMA4:    " << std::boolalpha << HAS_FMA4;
  LOG(INFO) << "Has AVX:     " << std::boolalpha << HAS_AVX;
  LOG(INFO) << "Has AVX2:    " << std::boolalpha << HAS_AVX2;
  LOG(INFO) << "Has AVX512:  " << std::boolalpha << HAS_AVX512;
  LOG(INFO) << "Has NEON:    " << std::boolalpha << HAS_NEON;
}
