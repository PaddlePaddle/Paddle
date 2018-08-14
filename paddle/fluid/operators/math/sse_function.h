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

#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace paddle {
namespace operators {
namespace math {

static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int SSE_CUT_LEN_MASK = 3U;
#define __m256x __m256
#define __m128x __m128
#define _mm_load_px _mm_loadu_ps
#define _mm_load1_px _mm_load1_ps
#define _mm_store_px _mm_storeu_ps
#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps

template <typename T>
inline void paddle_sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj,
                 _mm_add_px(_mm_load_px(y + jjj),
                            _mm_mul_px(mm_alpha, _mm_load_px(x + jjj))));
  }
  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}

template <typename T>
inline T paddle_uniform_real(T min, T max) {
  return ((T)rand() / (RAND_MAX)) * (max - min) + min;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
