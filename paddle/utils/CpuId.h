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

#pragma once

#include "Common.h"
#include "Error.h"

namespace paddle {

// clang-format off
enum simd_t {
  SIMD_NONE   = 0,          ///< None
  SIMD_SSE    = 1 << 0,     ///< SSE
  SIMD_SSE2   = 1 << 1,     ///< SSE 2
  SIMD_SSE3   = 1 << 2,     ///< SSE 3
  SIMD_SSSE3  = 1 << 3,     ///< SSSE 3
  SIMD_SSE41  = 1 << 4,     ///< SSE 4.1
  SIMD_SSE42  = 1 << 5,     ///< SSE 4.2
  SIMD_FMA3   = 1 << 6,     ///< FMA 3
  SIMD_FMA4   = 1 << 7,     ///< FMA 4
  SIMD_AVX    = 1 << 8,     ///< AVX
  SIMD_AVX2   = 1 << 9,     ///< AVX 2
  SIMD_AVX512 = 1 << 10,    ///< AVX 512
  SIMD_NEON   = 1 << 11,    ///  NEON
};
// clang-format on

class SIMDFlags final {
public:
  DISABLE_COPY(SIMDFlags);

  SIMDFlags();

  static SIMDFlags const* instance();

  inline bool check(int flags) const {
    return !((simd_flags_ & flags) ^ flags);
  }

private:
  int simd_flags_ = SIMD_NONE;
};

/**
 * @brief   Check SIMD flags at runtime.
 *
 * For example.
 * @code{.cpp}
 *
 * if (HAS_SIMD(SIMD_AVX2 | SIMD_FMA4)) {
 *      avx2_fm4_stub();
 * } else if (HAS_SIMD(SIMD_AVX)) {
 *      avx_stub();
 * }
 *
 * @endcode
 */
#define HAS_SIMD(__flags) SIMDFlags::instance()->check(__flags)

/**
 * @brief   Check SIMD flags at runtime.
 *
 * 1. Check all SIMD flags at runtime:
 *
 * @code{.cpp}
 * if (HAS_AVX && HAS_AVX2) {
 *      avx2_stub();
 * }
 * @endcod
 *
 * 2. Check one SIMD flag at runtime:
 *
 * @code{.cpp}
 * if (HAS_SSE41 || HAS_SSE42) {
 *      sse4_stub();
 * }
 * @endcode
 */
// clang-format off
#define HAS_SSE     HAS_SIMD(SIMD_SSE)
#define HAS_SSE2    HAS_SIMD(SIMD_SSE2)
#define HAS_SSE3    HAS_SIMD(SIMD_SSE3)
#define HAS_SSSE3   HAS_SIMD(SIMD_SSSE3)
#define HAS_SSE41   HAS_SIMD(SIMD_SSE41)
#define HAS_SSE42   HAS_SIMD(SIMD_SSE42)
#define HAS_FMA3    HAS_SIMD(SIMD_FMA3)
#define HAS_FMA4    HAS_SIMD(SIMD_FMA4)
#define HAS_AVX     HAS_SIMD(SIMD_AVX)
#define HAS_AVX2    HAS_SIMD(SIMD_AVX2)
#define HAS_AVX512  HAS_SIMD(SIMD_AVX512)
#define HAS_NEON    HAS_SIMD(SIMD_NEON)
// clang-format on

/**
 * Invoke checkCPUFeature() before Paddle initialization to
 * check target machine whether support compiled instructions.
 * If not, simply throw out an error.
 */
inline Error __must_check checkCPUFeature() {
  Error err;
#ifndef __AVX__
  if (HAS_AVX) {
    LOG(WARNING) << "PaddlePaddle wasn't compiled to use avx instructions, "
                 << "but these are available on your machine and could "
                 << "speed up CPU computations via CMAKE .. -DWITH_AVX=ON";
  }
#else
  if (!HAS_AVX) {
    err = Error(
        "PaddlePaddle was compiled to use avx instructions, "
        "but these aren't available on your machine, please "
        "disable it via CMAKE .. -DWITH_AVX=OFF");
  }
#endif  // __AVX__
#ifdef __SSE3__
  if (!HAS_SSE3) {
    err = Error(
        "PaddlePaddle was compiled to use sse3 instructions, "
        "which is the minimum requirement of PaddlePaddle. "
        "But these aren't available on your current machine.");
  }
#endif  // __SSE3__

  return err;
}

}  // namespace paddle
