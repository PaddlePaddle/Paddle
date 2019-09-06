/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <immintrin.h>  // sse
#include <cfloat>
#include <cmath>    //fabs
#include <cstring>  // memcpy

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/mklml.h"
//#include "naive_gemm.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
void call_gemm(const math::BlasT<DeviceContext, T>& blas,
               const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const T alpha, const T* A,
               const T* B, const T beta, T* C) {
#ifndef __NAIVE_GEMM__
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
#else
  naive::gemm((TransA == CblasTrans), (TransB == CblasTrans), M, N, K, alpha, A,
              B, beta, C);
#endif  // !__NAIVE_GEMM__
}

template <typename T>
void call_gemm(const framework::ExecutionContext& ctx,
               const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const T alpha, const T* A,
               const T* B, const T beta, T* C) {
#ifndef __NAIVE_GEMM__
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
#else
  naive::gemm((TransA == CblasTrans), (TransB == CblasTrans), M, N, K, alpha, A,
              B, beta, C);
#endif  // !__NAIVE_GEMM__
}

template <typename T>
void call_gemm_batched(const framework::ExecutionContext& ctx,
                       const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const T alpha, const T** A,
                       const T** B, const T beta, T** C,
                       const int batch) {
  for (int i = 0; i < batch; ++i) {
    call_gemm(ctx, TransA, TransB, M, N, K, alpha, A[i], B[i], beta, C[i]);
  }
}

// To align with Lego
#ifndef LEGO_USE_FLOAT
#define LEGO_USE_FLOAT
#endif
#ifndef LEGO_SSE
#define LEGO_SSE
#endif

#if defined(LEGO_USE_FLOAT)

#define __m256x __m256
#define __m128x __m128

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int AVX_CUT_LEN_MASK = 7U;
static const unsigned int SSE_CUT_LEN_MASK = 3U;

#define _mm256_setzero_px _mm256_setzero_ps
#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_hadd_px _mm256_hadd_ps
#define _mm256_permute2f128_px _mm256_permute2f128_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss
#define _mm256_castpx256_px128 _mm256_castps256_ps128
#define _mm256_max_px _mm256_max_ps
#define _mm256_sub_px _mm256_sub_ps
#define _mm256_set1_px _mm256_set1_ps
#define _mm256_sqrt_px _mm256_sqrt_ps
#define _mm256_div_px _mm256_div_ps
#define _mm_setzero_px _mm_setzero_ps
#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps
#define _mm_load_px _mm_loadu_ps
#define _mm_hadd_px _mm_hadd_ps
#define _mm_store_sx _mm_store_ss
#define _mm_store_px _mm_storeu_ps
#define _mm_load1_px _mm_load1_ps
#define _mm_max_px _mm_max_ps
#define _mm_sub_px _mm_sub_ps
#define _mm_set1_px _mm_set1_ps
#define _mm_sqrt_px _mm_sqrt_ps
#define _mm_div_px _mm_div_ps

#elif defined(LEGO_USE_DOUBLE)

#define __m256x __m256d
#define __m128x __m128d

static const unsigned int AVX_STEP_SIZE = 4;
static const unsigned int SSE_STEP_SIZE = 2;
static const unsigned int AVX_CUT_LEN_MASK = 3U;
static const unsigned int SSE_CUT_LEN_MASK = 1U;

#define _mm256_setzero_px _mm256_setzero_pd
#define _mm256_mul_px _mm256_mul_pd
#define _mm256_add_px _mm256_add_pd
#define _mm256_load_px _mm256_loadu_pd
#define _mm256_hadd_px _mm256_hadd_pd
#define _mm256_permute2f128_px _mm256_permute2f128_pd
#define _mm256_store_px _mm256_storeu_pd
#define _mm256_broadcast_sx _mm256_broadcast_sd
#define _mm256_castpx256_px128 _mm256_castpd256_pd128
#define _mm256_max_px _mm256_max_pd
#define _mm256_sub_px _mm256_sub_pd
#define _mm256_set1_px _mm256_set1_pd
#define _mm256_sqrt_px _mm256_sqrt_pd
#define _mm256_div_px _mm256_div_pd
#define _mm_setzero_px _mm_setzero_pd
#define _mm_add_px _mm_add_pd
#define _mm_mul_px _mm_mul_pd
#define _mm_load_px _mm_loadu_pd
#define _mm_hadd_px _mm_hadd_pd
#define _mm_store_sx _mm_store_sd
#define _mm_store_px _mm_storeu_pd
#define _mm_load1_px _mm_load1_pd
#define _mm_max_px _mm_max_pd
#define _mm_sub_px _mm_sub_pd
#define _mm_set1_px _mm_set1_pd
#define _mm_sqrt_px _mm_sqrt_pd
#define _mm_div_px _mm_div_pd
#endif

#if defined(LEGO_USE_FLOAT)

#define X_MIN FLT_MIN
#define X_MAX FLT_MAX

#elif defined(LEGO_USE_DOUBLE)

#define X_MIN DBL_MIN
#define X_MAX DBL_MAX

#endif

template <typename T>
inline void sse_eltadd(const T* x, const T* y, T* z, size_t len) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(z + jjj, _mm256_add_px(_mm256_load_px(x + jjj),
                                           _mm256_load_px(y + jjj)));
  }
#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;

  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(z + jjj,
                 _mm_add_px(_mm_load_px(x + jjj), _mm_load_px(y + jjj)));
  }
#endif
  for (; jjj < len; jjj++) {
    z[jjj] = x[jjj] + y[jjj];
  }
}

template <typename T>
inline void sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj,
                 _mm_add_px(_mm_load_px(y + jjj),
                            _mm_mul_px(mm_alpha, _mm_load_px(x + jjj))));
  }

#endif
  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}

template <typename T>
inline void sse_axpy_noadd(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(y + jjj, _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj)));
  }

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj, _mm_mul_px(mm_alpha, _mm_load_px(x + jjj)));
  }

#endif
  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}

template <typename T>
inline void sse_eltmul(const T* x, const T* y, T* z, size_t len) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(z + jjj, _mm256_mul_px(_mm256_load_px(x + jjj),
                                           _mm256_load_px(y + jjj)));
  }
#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;

  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(z + jjj,
                 _mm_mul_px(_mm_load_px(x + jjj), _mm_load_px(y + jjj)));
  }
#endif
  for (; jjj < len; jjj++) {
    z[jjj] = x[jjj] * y[jjj];
  }
}

template <typename T>
inline void sse_add_scalar(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(y + jjj, _mm256_add_px(mm_alpha, _mm256_load_px(x + jjj)));
  }

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj, _mm_add_px(mm_alpha, _mm_load_px(x + jjj)));
  }

#endif
  for (; jjj < len; jjj++) {
    y[jjj] = alpha + x[jjj];
  }
}

template <typename T>
inline void sse_sum(const T* x, T& y, size_t len) {
  unsigned int jjj, lll;
  jjj = lll = 0;
  y = 0.;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;

  __m256x mm_result = _mm256_setzero_px();
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    mm_result = _mm256_add_px(mm_result, _mm256_load_px(x + jjj));
  }

#if defined(LEGO_USE_FLOAT)
  __m256x hsum = _mm256_hadd_px(mm_result, mm_result);
#elif defined(LEGO_USE_DOUBLE)
  __m256x hsum = mm_result;
#endif
  hsum = _mm256_add_px(hsum, _mm256_permute2f128_px(hsum, hsum, 0x1));
  _mm_store_sx(&y, _mm_hadd_px(_mm256_castpx256_px128(hsum),
                               _mm256_castpx256_px128(hsum)));

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;

  __m128x mm_result = _mm_setzero_px();
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    mm_result = _mm_add_px(mm_result, _mm_load_px(x + jjj));
  }
  __m128x mm_tmp = _mm_hadd_px(mm_result, mm_result);

#if defined(LEGO_USE_FLOAT)
  _mm_store_sx(&y, _mm_hadd_px(mm_tmp, mm_tmp));
#elif defined(LEGO_USE_DOUBLE)
  _mm_store_sx(&y, mm_tmp);
#endif

#endif
  for (; jjj < len; jjj++) {
    y += x[jjj];
  }
}

template <typename T>
inline void sse_scale(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);

  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(y + jjj, _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj)));
  }

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj, _mm_mul_px(mm_alpha, _mm_load_px(x + jjj)));
  }
#endif
  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}

template <typename T>
inline void sse_ip(const T* vec1, const T* vec2, size_t len, T& result) {
  unsigned int jjj, lll;
  jjj = lll = 0;
  result = 0.;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;

  __m256x mm_result = _mm256_setzero_px();
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    mm_result = _mm256_add_px(
        mm_result,
        _mm256_mul_px(_mm256_load_px(vec1 + jjj), _mm256_load_px(vec2 + jjj)));
  }

  //    result = mm_result[0]+mm_result[1]+mm_result[2]+mm_result[3]+
  //      mm_result[4]+mm_result[5]+mm_result[6]+mm_result[7];

#if defined(LEGO_USE_FLOAT)
  __m256x hsum = _mm256_hadd_px(mm_result, mm_result);
#elif defined(LEGO_USE_DOUBLE)
  __m256x hsum = mm_result;
#endif

  hsum = _mm256_add_px(hsum, _mm256_permute2f128_px(hsum, hsum, 0x1));

  _mm_store_sx(&result, _mm_hadd_px(_mm256_castpx256_px128(hsum),
                                    _mm256_castpx256_px128(hsum)));

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_result = _mm_setzero_px();
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    mm_result = _mm_add_px(mm_result, _mm_mul_px(_mm_load_px(vec1 + jjj),
                                                 _mm_load_px(vec2 + jjj)));
  }
  __m128x mm_tmp = _mm_hadd_px(mm_result, mm_result);
#if defined(LEGO_USE_FLOAT)
  _mm_store_sx(&result, _mm_hadd_px(mm_tmp, mm_tmp));
#elif defined(LEGO_USE_DOUBLE)
  _mm_store_sx(&result, mm_tmp);
#endif

#endif
  for (; jjj < len; jjj++) {
    result += vec1[jjj] * vec2[jjj];
  }
}

}  // namespace operators
}  // namespace paddle
