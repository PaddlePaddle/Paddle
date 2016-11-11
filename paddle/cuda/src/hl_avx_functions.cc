/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include <immintrin.h>
#include "hl_functions.h"

namespace hppl {

  extern __m256 exp(__m256 a);

  __m256 relu(const __m256 a) {
    __m256 tmp = _mm256_set1_ps(0.0f);
    return _mm256_max_ps(a, tmp);
  }

  __m256 sigmoid(const __m256 a) {
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
    __m256 tmp = _mm256_max_ps(a, min);
    tmp = _mm256_min_ps(tmp, max);
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = exp(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
  }

  __m256 tanh(const __m256 a) {
    __m256 max = _mm256_set1_ps(EXP_MAX_INPUT);
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = _mm256_min_ps(tmp, max);
    tmp = exp(tmp);
    return _mm256_sub_ps(
        _mm256_div_ps(_mm256_set1_ps(2.0f),
        _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)), _mm256_set1_ps(1.0f));
  }

  __m256 linear(const __m256 a) {
    return a;
  }

  __m256 relu(const __m256 a, const __m256 b) {
    return _mm256_mul_ps(a,
      _mm256_and_ps(_mm256_cmp_ps(b, _mm256_set1_ps(0.0f), _CMP_GT_OS),
      _mm256_set1_ps(1.0f)));
  }

  __m256 sigmoid(const __m256 a, const __m256 b) {
    return _mm256_mul_ps(_mm256_mul_ps(a, b),
        _mm256_sub_ps(_mm256_set1_ps(1.0f), b));
  }

  __m256 tanh(const __m256 a, const __m256 b) {
    return _mm256_mul_ps(a,
      _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(b, b)));
  }

  __m256 linear(const __m256 a, const __m256 b) {
    return a;
  }
}  // namespace hppl
