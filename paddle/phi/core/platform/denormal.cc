// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/denormal.h"

#include <tuple>
#include <utility>

// Refer to https://github.com/tensorflow/tensorflow/pull/17141

// If we're on gcc 4.8 or older, there's a known bug that prevents the use of
// intrinsics when the architecture is not defined in the flags. See
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57202
#if !defined(__SSE3__) && !defined(__clang__) && \
    (defined(__GNUC__) && (__GNUC__ < 4) ||      \
     ((__GNUC__ == 4) && (__GNUC_MINOR__ < 9)))
#define GCC_WITHOUT_INTRINSICS
#endif

#if !defined(GCC_WITHOUT_INTRINSICS) && !defined(PADDLE_WITH_ARM) && \
    !defined(PADDLE_WITH_SW) && !defined(PADDLE_WITH_MIPS) &&        \
    !defined(_WIN32) && !defined(PADDLE_WITH_LOONGARCH)
#define DENORM_USE_INTRINSICS
#endif

#ifdef DENORM_USE_INTRINSICS
#include <pmmintrin.h>
#endif

namespace paddle::platform {

static void SetDenormalState(bool flush_zero_mode, bool denormals_zero_mode) {
#ifdef DENORM_USE_INTRINSICS
#ifdef PADDLE_WITH_SSE3
  // Intel's C and Fortran compilers enable the denormals-are-zero (DAZ) and
  // flush-to-zero (FTZ) flags for SSE by default for optimization levels higher
  // than -O0.
  // AArch32 NEON (SIMD) FPU always uses a flush-to-zero mode.
  // Refer to https://en.wikipedia.org/wiki/Denormal_number
  // and https://software.intel.com/sites/landingpage/IntrinsicsGuide/
  _MM_SET_FLUSH_ZERO_MODE(flush_zero_mode ? _MM_FLUSH_ZERO_ON
                                          : _MM_FLUSH_ZERO_OFF);
  _MM_SET_DENORMALS_ZERO_MODE(denormals_zero_mode ? _MM_DENORMALS_ZERO_ON
                                                  : _MM_DENORMALS_ZERO_OFF);
#endif
#endif
}

static std::pair<bool, bool> GetDenormalState() {
#ifdef DENORM_USE_INTRINSICS
#ifdef PADDLE_WITH_SSE3
  bool flush_zero_mode = _MM_GET_FLUSH_ZERO_MODE() == _MM_FLUSH_ZERO_ON;
  bool denormals_zero_mode =
      _MM_GET_DENORMALS_ZERO_MODE() == _MM_DENORMALS_ZERO_ON;
  return {flush_zero_mode, denormals_zero_mode};
#endif
#endif
  return {false, false};
}

ScopedRestoreFlushDenormalState::ScopedRestoreFlushDenormalState()
    : flush_zero_mode_(false), denormals_zero_mode_(false) {
  std::tie(flush_zero_mode_, denormals_zero_mode_) = GetDenormalState();
}

ScopedRestoreFlushDenormalState::~ScopedRestoreFlushDenormalState() {
  SetDenormalState(flush_zero_mode_, denormals_zero_mode_);
}

ScopedFlushDenormal::ScopedFlushDenormal() { SetDenormalState(true, true); }
}  // namespace paddle::platform
