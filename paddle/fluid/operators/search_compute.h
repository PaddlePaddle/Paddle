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

#define __m256x __m256

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int AVX_CUT_LEN_MASK = 7U;

#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss

#define _mm256_mul_pd _mm256_mul_pd
#define _mm256_add_pd _mm256_add_pd
#define _mm256_load_pd _mm256_loadu_pd
#define _mm256_store_pd _mm256_storeu_pd
#define _mm256_broadcast_sd _mm256_broadcast_sd

inline void avx_axpy(const float* x, float* y, size_t len, const float alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }

  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}

inline void avx_axpy(const double* x, double* y, size_t len,
                     const float alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~AVX_CUT_LEN_MASK;
  double alpha_d = static_cast<double>(alpha);

  __m256d mm_alpha = _mm256_broadcast_sd(&alpha_d);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_pd(
        y + jjj,
        _mm256_add_pd(_mm256_load_pd(y + jjj),
                      _mm256_mul_pd(mm_alpha, _mm256_load_pd(x + jjj))));
  }

  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}
inline void avx_axpy_noadd(const double* x, double* y, size_t len,
                           const float alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;
  double alpha_d = static_cast<double>(alpha);
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256d mm_alpha = _mm256_broadcast_sd(&alpha_d);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_pd(y + jjj, _mm256_mul_pd(mm_alpha, _mm256_load_pd(x + jjj)));
  }

  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}
inline void avx_axpy_noadd(const float* x, float* y, size_t len,
                           const float alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(y + jjj, _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj)));
  }

  for (; jjj < len; jjj++) {
    y[jjj] = alpha * x[jjj];
  }
}
inline void avx_axpy_noadd(const int8_t* x, int8_t* y, size_t len,
                           const float alpha) {
  PADDLE_THROW(platform::errors::Unimplemented(
      "int8_t input of avx_axpy_noadd is  not supported"));
}

}  // namespace operators
}  // namespace paddle
