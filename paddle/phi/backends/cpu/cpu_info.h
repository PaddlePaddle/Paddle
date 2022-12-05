// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <stddef.h>

#ifdef _WIN32
#if defined(__AVX2__)
#include <immintrin.h>  // avx2
#elif defined(__AVX__)
#include <intrin.h>  // avx
#endif               // AVX
#else                // WIN32
#ifdef __AVX__
#include <immintrin.h>
#endif
#endif  // WIN32

#if defined(_WIN32)
#define ALIGN32_BEG __declspec(align(32))
#define ALIGN32_END
#else
#define ALIGN32_BEG
#define ALIGN32_END __attribute__((aligned(32)))
#endif  // _WIN32

namespace phi {
namespace backends {
namespace cpu {

//! Get the minimum chunk size for buddy allocator.
inline size_t CpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 4 KB.
  return 1 << 12;
}

typedef enum {
  isa_any,
  sse42,
  avx,
  avx2,
  avx512f,
  avx512_core,
  avx512_core_vnni,
  avx512_mic,
  avx512_mic_4ops,
  avx512_bf16,
} cpu_isa_t;  // Instruction set architecture

}  // namespace cpu
}  // namespace backends
}  // namespace phi
