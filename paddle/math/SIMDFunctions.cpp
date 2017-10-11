/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "SIMDFunctions.h"
#ifdef __SSE3__
#include <immintrin.h>
#endif
#include <algorithm>

#ifdef __AVX__
static void addto_avx(float* a, const float* b, size_t len) {
  int offset = len % 32;

  __m256 ma0, ma1, ma2, ma3;
  __m256 mb0, mb1, mb2, mb3;

  for (unsigned int k = 0; k < len / 32; k++, a += 32, b += 32) {
    ma0 = _mm256_load_ps(a);
    ma1 = _mm256_load_ps(a + 8);
    ma2 = _mm256_load_ps(a + 16);
    ma3 = _mm256_load_ps(a + 24);

    mb0 = _mm256_load_ps(b);
    mb1 = _mm256_load_ps(b + 8);
    mb2 = _mm256_load_ps(b + 16);
    mb3 = _mm256_load_ps(b + 24);

    ma0 = _mm256_add_ps(ma0, mb0);
    ma1 = _mm256_add_ps(ma1, mb1);
    ma2 = _mm256_add_ps(ma2, mb2);
    ma3 = _mm256_add_ps(ma3, mb3);

    _mm256_store_ps(a, ma0);
    _mm256_store_ps(a + 8, ma1);
    _mm256_store_ps(a + 16, ma2);
    _mm256_store_ps(a + 24, ma3);
  }

  for (int i = 0; i < offset; i++) a[i] += b[i];

  return;
}

static void batch_addto_avx(float* a, const float* b[], int batch, size_t len) {
  int offset = len % 32;

  __m256 ma0, ma1, ma2, ma3;
  __m256 mb0, mb1, mb2, mb3;

  for (unsigned int k = 0; k < len / 32; k++, a += 32) {
    ma0 = _mm256_load_ps(a);
    ma1 = _mm256_load_ps(a + 8);
    ma2 = _mm256_load_ps(a + 16);
    ma3 = _mm256_load_ps(a + 24);

    for (int i = 0; i < batch; i++) {
      mb0 = _mm256_load_ps(b[i]);
      mb1 = _mm256_load_ps(b[i] + 8);
      mb2 = _mm256_load_ps(b[i] + 16);
      mb3 = _mm256_load_ps(b[i] + 24);
      ma0 = _mm256_add_ps(ma0, mb0);
      ma1 = _mm256_add_ps(ma1, mb1);
      ma2 = _mm256_add_ps(ma2, mb2);
      ma3 = _mm256_add_ps(ma3, mb3);
      b[i] += 32;
    }

    _mm256_store_ps(a, ma0);
    _mm256_store_ps(a + 8, ma1);
    _mm256_store_ps(a + 16, ma2);
    _mm256_store_ps(a + 24, ma3);
  }

  for (int i = 0; i < offset; i++) {
    for (int k = 0; k < batch; k++) a[i] += b[k][i];
  }
  return;
}

static void col_max_avx(float* result,
                        const float* data,
                        int dim,
                        int numSamples) {
  // first sample, direct copy
  for (int d = 0; d < dim; ++d) {
    result[d] = data[d];
  }
  int offset = dim % 32;
  __m256 ma0, ma1, ma2, ma3;
  __m256 mb0, mb1, mb2, mb3;
  // first 16n dims
  for (int k = 0; k < dim / 32; k++, result += 32, data += 32) {
    ma0 = _mm256_load_ps(result);
    ma1 = _mm256_load_ps(result + 8);
    ma2 = _mm256_load_ps(result + 16);
    ma3 = _mm256_load_ps(result + 24);
    for (int i = 1; i < numSamples; i++) {
      mb0 = _mm256_load_ps(data + i * dim);
      mb1 = _mm256_load_ps(data + i * dim + 8);
      mb2 = _mm256_load_ps(data + i * dim + 16);
      mb3 = _mm256_load_ps(data + i * dim + 24);
      ma0 = _mm256_max_ps(ma0, mb0);
      ma1 = _mm256_max_ps(ma1, mb1);
      ma2 = _mm256_max_ps(ma2, mb2);
      ma3 = _mm256_max_ps(ma3, mb3);
    }
    _mm256_store_ps(result, ma0);
    _mm256_store_ps(result + 8, ma1);
    _mm256_store_ps(result + 16, ma2);
    _mm256_store_ps(result + 24, ma3);
  }
  // last dims
  for (int d = 0; d < offset; ++d) {
    float sm = data[d];
    for (int i = 1; i < numSamples; ++i) {
      sm = std::max(sm, data[i * dim + d]);
    }
    result[d] = sm;
  }
}

static void decayL1_avx(float* dst, float* src, float lambda, size_t sz) {
  int64_t i;
  int64_t size = sz;
  float src_val;

  __m256 ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8;
  //  __m256 ymm9, ymm10;

  ymm1 = _mm256_set1_ps(lambda);
  ymm2 = _mm256_setzero_ps();

  for (i = 0; i <= size - 16; i += 16) {
    ymm3 = _mm256_load_ps(src + i);
    ymm6 = _mm256_load_ps(src + i + 8);

    ymm4 = _mm256_sub_ps(ymm3, ymm1);
    ymm7 = _mm256_sub_ps(ymm6, ymm1);

    ymm5 = _mm256_add_ps(ymm3, ymm1);
    ymm8 = _mm256_add_ps(ymm6, ymm1);

    ymm4 = _mm256_max_ps(ymm4, ymm2);
    ymm7 = _mm256_max_ps(ymm7, ymm2);

    ymm5 = _mm256_min_ps(ymm5, ymm2);
    ymm8 = _mm256_min_ps(ymm8, ymm2);

    ymm5 = _mm256_or_ps(ymm4, ymm5);
    ymm8 = _mm256_or_ps(ymm7, ymm8);

    _mm256_store_ps(dst + i, ymm5);
    _mm256_store_ps(dst + i + 8, ymm8);
  }
  if (i <= size - 8) {
    ymm3 = _mm256_load_ps(src + i);
    ymm4 = _mm256_sub_ps(ymm3, ymm1);
    ymm5 = _mm256_add_ps(ymm3, ymm1);
    ymm4 = _mm256_max_ps(ymm4, ymm2);
    ymm5 = _mm256_min_ps(ymm5, ymm2);
    ymm5 = _mm256_or_ps(ymm4, ymm5);
    _mm256_store_ps(dst + i, ymm5);

    i += 8;
  }
  for (; i < size; i++) {
    src_val = src[i];
    if (src_val > 0) {
      dst[i] = ((src_val > lambda) ? (src_val - lambda) : 0);
    } else {
      dst[i] = ((-src_val > lambda) ? (src_val + lambda) : 0);
    }
  }
}

static void decayL1_avx(
    float* dst, float* src, float* lr, float lambda, size_t sz) {
  int64_t i;
  int64_t size = sz;
  float src_val;

  __m256 ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8;
  __m256 ymm9, ymm10;

  ymm1 = _mm256_set1_ps(lambda);
  ymm2 = _mm256_setzero_ps();

  for (i = 0; i <= size - 16; i += 16) {
    ymm9 = _mm256_load_ps(lr + i);
    ymm10 = _mm256_load_ps(lr + i + 8);

    ymm3 = _mm256_load_ps(src + i);
    ymm6 = _mm256_load_ps(src + i + 8);

    ymm9 = _mm256_mul_ps(ymm9, ymm1);
    ymm10 = _mm256_mul_ps(ymm10, ymm1);

    ymm4 = _mm256_sub_ps(ymm3, ymm9);
    ymm7 = _mm256_sub_ps(ymm6, ymm10);

    ymm5 = _mm256_add_ps(ymm3, ymm9);
    ymm8 = _mm256_add_ps(ymm6, ymm10);

    ymm4 = _mm256_max_ps(ymm4, ymm2);
    ymm7 = _mm256_max_ps(ymm7, ymm2);

    ymm5 = _mm256_min_ps(ymm5, ymm2);
    ymm8 = _mm256_min_ps(ymm8, ymm2);

    ymm5 = _mm256_or_ps(ymm4, ymm5);
    ymm8 = _mm256_or_ps(ymm7, ymm8);

    _mm256_store_ps(dst + i, ymm5);
    _mm256_store_ps(dst + i + 8, ymm8);
  }
  if (i <= size - 8) {
    ymm3 = _mm256_load_ps(src + i);
    ymm9 = _mm256_load_ps(lr + i);
    ymm9 = _mm256_mul_ps(ymm9, ymm1);
    ymm4 = _mm256_sub_ps(ymm3, ymm9);
    ymm5 = _mm256_add_ps(ymm3, ymm9);
    ymm4 = _mm256_max_ps(ymm4, ymm2);
    ymm5 = _mm256_min_ps(ymm5, ymm2);
    ymm5 = _mm256_or_ps(ymm4, ymm5);
    _mm256_store_ps(dst + i, ymm5);

    i += 8;
  }
  for (; i < size; i++) {
    src_val = src[i];
    float nlambda = lr[i] * lambda;
    if (src_val > 0) {
      dst[i] = ((src_val > nlambda) ? (src_val - nlambda) : 0);
    } else {
      dst[i] = ((-src_val > nlambda) ? (src_val + nlambda) : 0);
    }
  }
}

#elif defined(__SSE3__)

static void addto_sse(float* a, const float* b, size_t len) {
  int offset = len % 16;
  __m128 ma0, ma1, ma2, ma3;
  __m128 mb0, mb1, mb2, mb3;

  for (unsigned int k = 0; k < len / 16; k++, a += 16, b += 16) {
    ma0 = _mm_load_ps(a);
    ma1 = _mm_load_ps(a + 4);
    ma2 = _mm_load_ps(a + 8);
    ma3 = _mm_load_ps(a + 12);

    mb0 = _mm_load_ps(b);
    mb1 = _mm_load_ps(b + 4);
    mb2 = _mm_load_ps(b + 8);
    mb3 = _mm_load_ps(b + 12);

    ma0 = _mm_add_ps(ma0, mb0);
    ma1 = _mm_add_ps(ma1, mb1);
    ma2 = _mm_add_ps(ma2, mb2);
    ma3 = _mm_add_ps(ma3, mb3);

    _mm_store_ps(a, ma0);
    _mm_store_ps(a + 4, ma1);
    _mm_store_ps(a + 8, ma2);
    _mm_store_ps(a + 12, ma3);
  }

  for (int i = 0; i < offset; i++) a[i] += b[i];
}

static void batch_addto_sse(float* a, const float* b[], int batch, size_t len) {
  int offset = len % 16;

  __m128 ma0, ma1, ma2, ma3;
  __m128 mb0, mb1, mb2, mb3;

  for (unsigned int k = 0; k < len / 16; k++, a += 16) {
    ma0 = _mm_load_ps(a);
    ma1 = _mm_load_ps(a + 4);
    ma2 = _mm_load_ps(a + 8);
    ma3 = _mm_load_ps(a + 12);

    for (int i = 0; i < batch; i++) {
      mb0 = _mm_load_ps(b[i]);
      mb1 = _mm_load_ps(b[i] + 4);
      mb2 = _mm_load_ps(b[i] + 8);
      mb3 = _mm_load_ps(b[i] + 12);
      ma0 = _mm_add_ps(ma0, mb0);
      ma1 = _mm_add_ps(ma1, mb1);
      ma2 = _mm_add_ps(ma2, mb2);
      ma3 = _mm_add_ps(ma3, mb3);
      b[i] += 16;
    }

    _mm_store_ps(a, ma0);
    _mm_store_ps(a + 4, ma1);
    _mm_store_ps(a + 8, ma2);
    _mm_store_ps(a + 12, ma3);
  }

  for (int i = 0; i < offset; i++) {
    for (int k = 0; k < batch; k++) a[i] += b[k][i];
  }
  return;
}

static void col_max_sse(float* result,
                        const float* data,
                        int dim,
                        int numSamples) {
  // first sample, direct copy
  for (int d = 0; d < dim; ++d) {
    result[d] = data[d];
  }
  int offset = dim % 16;
  __m128 ma0, ma1, ma2, ma3;
  __m128 mb0, mb1, mb2, mb3;
  // first 16n dims
  for (int k = 0; k < dim / 16; k++, result += 16, data += 16) {
    ma0 = _mm_load_ps(result);
    ma1 = _mm_load_ps(result + 4);
    ma2 = _mm_load_ps(result + 8);
    ma3 = _mm_load_ps(result + 12);
    for (int i = 1; i < numSamples; i++) {
      mb0 = _mm_load_ps(data + i * dim);
      mb1 = _mm_load_ps(data + i * dim + 4);
      mb2 = _mm_load_ps(data + i * dim + 8);
      mb3 = _mm_load_ps(data + i * dim + 12);
      ma0 = _mm_max_ps(ma0, mb0);
      ma1 = _mm_max_ps(ma1, mb1);
      ma2 = _mm_max_ps(ma2, mb2);
      ma3 = _mm_max_ps(ma3, mb3);
    }
    _mm_store_ps(result, ma0);
    _mm_store_ps(result + 4, ma1);
    _mm_store_ps(result + 8, ma2);
    _mm_store_ps(result + 12, ma3);
  }
  // last dims
  for (int d = 0; d < offset; ++d) {
    float sm = data[d];
    for (int i = 1; i < numSamples; ++i) {
      sm = std::max(sm, data[i * dim + d]);
    }
    result[d] = sm;
  }
}

#endif

#if defined(__AVX__)
#define SIMD_INVOKE(func, ...) func##_avx(__VA_ARGS__)
#elif defined(__SSE3__)
#define SIMD_INVOKE(func, ...) func##_sse(__VA_ARGS__)
#endif

namespace paddle {
namespace simd {
namespace internal {
#ifdef __SSE3__
void addToImpl(float* a, const float* b, size_t len) {
  SIMD_INVOKE(addto, a, b, len);
}
void batchAddToImpl(float* a, const float* b[], int batch, size_t len) {
  SIMD_INVOKE(batch_addto, a, b, batch, len);
}

void colMaxImpl(float* result, const float* data, int dim, int numSamples) {
  SIMD_INVOKE(col_max, result, data, dim, numSamples);
}
#endif

#ifdef __AVX__
void decayL1AvxImpl(float* dst, float* src, float lambda, size_t len) {
  decayL1_avx(dst, src, lambda, len);
}
void decayL1AvxImpl(
    float* dst, float* src, float* lr, float lambda, size_t len) {
  decayL1_avx(dst, src, lr, lambda, len);
}
#endif

}  // namespace internal
}  // namespace simd
}  // namespace paddle
