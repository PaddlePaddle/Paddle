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

template <typename T>
inline void paddle_sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128 mm_alpha = _mm_load1_ps(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_storeu_ps(y + jjj,
                  _mm_add_ps(_mm_loadu_ps(y + jjj),
                             _mm_mul_ps(mm_alpha, _mm_loadu_ps(x + jjj))));
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
