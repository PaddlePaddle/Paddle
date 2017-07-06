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

#include "paddle/utils/CpuId.h"
#include "paddle/utils/Util.h"

#ifdef _WIN32

#include <intrin.h>

/// for MSVC
#define CPUID(info, x) __cpuidex(info, x, 0)

#else

#if !defined(__arm__) && !defined(__aarch64__)
#include <cpuid.h>
/// for GCC/Clang
#define CPUID(info, x) __cpuid_count(x, 0, info[0], info[1], info[2], info[3])
#endif

#endif

namespace paddle {

SIMDFlags::SIMDFlags() {
#if defined(__arm__) || defined(__aarch64__)
  simd_flags_ = SIMD_NEON;
#else
  unsigned int cpuInfo[4];
  // CPUID: https://en.wikipedia.org/wiki/CPUID
  // clang-format off
  CPUID(cpuInfo, 0x00000001);
  simd_flags_ |= cpuInfo[3] & (1 << 25) ? SIMD_SSE   : SIMD_NONE;
  simd_flags_ |= cpuInfo[3] & (1 << 26) ? SIMD_SSE2  : SIMD_NONE;
  simd_flags_ |= cpuInfo[2] & (1 <<  0) ? SIMD_SSE3  : SIMD_NONE;
  simd_flags_ |= cpuInfo[2] & (1 <<  9) ? SIMD_SSSE3 : SIMD_NONE;
  simd_flags_ |= cpuInfo[2] & (1 << 19) ? SIMD_SSE41 : SIMD_NONE;
  simd_flags_ |= cpuInfo[2] & (1 << 20) ? SIMD_SSE42 : SIMD_NONE;
  simd_flags_ |= cpuInfo[2] & (1 << 12) ? SIMD_FMA3  : SIMD_NONE;
  simd_flags_ |= cpuInfo[2] & (1 << 28) ? SIMD_AVX   : SIMD_NONE;

  CPUID(cpuInfo, 0x00000007);
  simd_flags_ |= cpuInfo[1] & (1 <<  5) ? SIMD_AVX2  : SIMD_NONE;
  simd_flags_ |= cpuInfo[1] & (1 << 16) ? SIMD_AVX512: SIMD_NONE;

  CPUID(cpuInfo, 0x80000001);
  simd_flags_ |= cpuInfo[2] & (1 << 16) ? SIMD_FMA4  : SIMD_NONE;
  // clang-fotmat on
#endif
}

SIMDFlags const* SIMDFlags::instance() {
  static SIMDFlags instance;
  return &instance;
}

}  // namespace paddle
