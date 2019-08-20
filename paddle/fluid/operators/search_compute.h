/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <cfloat>
#include <cmath>
#include <cstring>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/mklml.h"

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
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename T>
void call_gemm(const framework::ExecutionContext& ctx,
               const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const T alpha, const T* A,
               const T* B, const T beta, T* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename DeviceContext, typename T>
void call_gemm_with_lda(const math::BlasT<DeviceContext, T>& blas,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const T alpha, const T* A, const T* B,
                        const T beta, T* C, int lda) {
  int ldb = (TransB == CblasNoTrans) ? N : K;

  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <typename T>
void call_gemm_batched(const framework::ExecutionContext& ctx,
                       const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB, const int M, const int N,
                       const int K, const T alpha, const T** A, const T** B,
                       const T beta, T** C, const int batch) {
  for (int i = 0; i < batch; ++i) {
    call_gemm(ctx, TransA, TransB, M, N, K, alpha, A[i], B[i], beta, C[i]);
  }
}

#ifndef TYPE_USE_FLOAT
#define TYPE_USE_FLOAT
#endif
#ifndef USE_SSE
#define USE_SSE
#endif

#if defined(TYPE_USE_FLOAT)

#define __m256x __m256
#define __m128x __m128

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int AVX_CUT_LEN_MASK = 7U;
static const unsigned int SSE_CUT_LEN_MASK = 3U;

#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss

#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps
#define _mm_load_px _mm_loadu_ps
#define _mm_store_px _mm_storeu_ps
#define _mm_load1_px _mm_load1_ps

#endif

template <typename T>
inline void sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(USE_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }

#elif defined(USE_SSE)
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

}  // namespace operators
}  // namespace paddle
