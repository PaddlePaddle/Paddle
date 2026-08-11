//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/cpu/cpu_context.h"
#ifdef PADDLE_WITH_MKLML
#include <mkl.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

namespace detail {
inline int to_blas_int(int64_t value, const char *name) {
  PADDLE_ENFORCE_GE(
      value,
      0,
      common::errors::InvalidArgument("BLAS parameter %s must be non-negative, "
                                      "but received %ld.",
                                      name,
                                      value));
  PADDLE_ENFORCE_LE_INT_MAX(value, name);
  return static_cast<int>(value);
}

template <typename T>
static void axpy_fallback(
    int64_t n, const T alpha, const T *x, int64_t incx, T *y, int64_t incy) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  const MT mp_alpha = static_cast<MT>(alpha);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t x_index = i * incx;
    const int64_t y_index = i * incy;
    y[y_index] = static_cast<T>(static_cast<MT>(y[y_index]) +
                                mp_alpha * static_cast<MT>(x[x_index]));
  }
}

inline bool level1_blas_compatible(int64_t n, int64_t incx, int64_t incy) {
  constexpr auto kIntMin = std::numeric_limits<int>::lowest();
  constexpr auto kIntMax = std::numeric_limits<int>::max();
  return n >= 0 && n <= kIntMax && incx >= kIntMin && incx <= kIntMax &&
         incy >= kIntMin && incy <= kIntMax;
}

template <typename T, typename BlasAxpy>
static void axpy_with_blas(int64_t n,
                           const T alpha,
                           const T *x,
                           int64_t incx,
                           T *y,
                           int64_t incy,
                           BlasAxpy blas_axpy) {
  if (n <= 0) {
    return;
  }
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  if (level1_blas_compatible(n, incx, incy)) {
    blas_axpy(static_cast<int>(n),
              alpha,
              x,
              static_cast<int>(incx),
              y,
              static_cast<int>(incy));
    return;
  }
  axpy_fallback(n, alpha, x, incx, y, incy);
}

template <typename T>
static void axpy(
    int64_t n, const T alpha, const T *x, int64_t incx, T *y, int64_t incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  axpy_fallback(n, alpha, x, incx, y, incy);
}

static void axpy(int64_t n,
                 const float alpha,
                 const float *x,
                 int64_t incx,
                 float *y,
                 int64_t incy) {
  axpy_with_blas(
      n,
      alpha,
      x,
      incx,
      y,
      incy,
      [](int n, float alpha, const float *x, int incx, float *y, int incy) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
        phi::dynload::cblas_saxpy(n, alpha, x, incx, y, incy);
#else
        cblas_saxpy(n, alpha, x, incx, y, incy);
#endif
      });
}

static void axpy(int64_t n,
                 const double alpha,
                 const double *x,
                 int64_t incx,
                 double *y,
                 int64_t incy) {
  axpy_with_blas(
      n,
      alpha,
      x,
      incx,
      y,
      incy,
      [](int n, double alpha, const double *x, int incx, double *y, int incy) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
        phi::dynload::cblas_daxpy(n, alpha, x, incx, y, incy);
#else
        cblas_daxpy(n, alpha, x, incx, y, incy);
#endif
      });
}

static void axpy(int64_t n,
                 const phi::complex64 alpha,
                 const phi::complex64 *x,
                 int64_t incx,
                 phi::complex64 *y,
                 int64_t incy) {
  axpy_with_blas(n,
                 alpha,
                 x,
                 incx,
                 y,
                 incy,
                 [](int n,
                    phi::complex64 alpha,
                    const phi::complex64 *x,
                    int incx,
                    phi::complex64 *y,
                    int incy) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
                   phi::dynload::cblas_caxpy(n, &alpha, x, incx, y, incy);
#else
        cblas_caxpy(n, &alpha, x, incx, y, incy);
#endif
                 });
}

static void axpy(int64_t n,
                 const phi::complex128 alpha,
                 const phi::complex128 *x,
                 int64_t incx,
                 phi::complex128 *y,
                 int64_t incy) {
  axpy_with_blas(n,
                 alpha,
                 x,
                 incx,
                 y,
                 incy,
                 [](int n,
                    phi::complex128 alpha,
                    const phi::complex128 *x,
                    int incx,
                    phi::complex128 *y,
                    int incy) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
                   phi::dynload::cblas_zaxpy(n, &alpha, x, incx, y, incy);
#else
        cblas_zaxpy(n, &alpha, x, incx, y, incy);
#endif
                 });
}

template <typename T>
static T dot_fallback(
    int64_t n, const T *x, int64_t incx, const T *y, int64_t incy) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  MT sum = static_cast<MT>(0);
  for (int64_t i = 0; i < n; ++i) {
    sum += static_cast<MT>(x[i * incx]) * static_cast<MT>(y[i * incy]);
  }
  return static_cast<T>(sum);
}

template <typename T, typename BlasDot>
static T dot_with_blas(int64_t n,
                       const T *x,
                       int64_t incx,
                       const T *y,
                       int64_t incy,
                       BlasDot blas_dot) {
  if (n <= 0) {
    return static_cast<T>(0);
  }
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  if (level1_blas_compatible(n, incx, incy)) {
    return blas_dot(static_cast<int>(n),
                    x,
                    static_cast<int>(incx),
                    y,
                    static_cast<int>(incy));
  }
  return dot_fallback(n, x, incx, y, incy);
}

template <typename T>
static T dot(int64_t n, const T *x, int64_t incx, const T *y, int64_t incy) {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  return dot_fallback(n, x, incx, y, incy);
}

static float dot(
    int64_t n, const float *x, int64_t incx, const float *y, int64_t incy) {
  return dot_with_blas(
      n,
      x,
      incx,
      y,
      incy,
      [](int n, const float *x, int incx, const float *y, int incy) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
        return phi::dynload::cblas_sdot(n, x, incx, y, incy);
#else
        return cblas_sdot(n, x, incx, y, incy);
#endif
      });
}

static double dot(
    int64_t n, const double *x, int64_t incx, const double *y, int64_t incy) {
  return dot_with_blas(
      n,
      x,
      incx,
      y,
      incy,
      [](int n, const double *x, int incx, const double *y, int incy) {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
        return phi::dynload::cblas_ddot(n, x, incx, y, incy);
#else
        return cblas_ddot(n, x, incx, y, incy);
#endif
      });
}
}  // namespace detail

template <typename T>
struct CBlas;

template <>
struct CBlas<int8_t> {};

template <>
struct CBlas<int16_t> {};

template <>
struct CBlas<phi::bfloat16> {
  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static phi::bfloat16 DOT(ARGS... args) {
    return detail::dot(args...);
  }
};

#ifdef PADDLE_WITH_MKLML
template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    phi::dynload::cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static float *GEMM_ALLOC(ARGS... args) {
    return phi::dynload::cblas_sgemm_alloc(args...);
  }

  template <typename... ARGS>
  static void GEMM_PACK(ARGS... args) {
    phi::dynload::cblas_sgemm_pack(args...);
  }

  template <typename... ARGS>
  static void GEMM_COMPUTE(ARGS... args) {
    phi::dynload::cblas_sgemm_compute(args...);
  }

  template <typename... ARGS>
  static void GEMM_FREE(ARGS... args) {
    phi::dynload::cblas_sgemm_free(args...);
  }

#ifdef PADDLE_WITH_LIBXSMM
  template <typename... ARGS>
  static void SMM_GEMM(ARGS... args) {
    libxsmm_sgemm(args...);
  }
#endif

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    phi::dynload::cblas_sgemv(args...);
  }

  template <typename... ARGS>
  static float DOT(ARGS... args) {
    return detail::dot(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    phi::dynload::cblas_sgemm_batch(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    phi::dynload::cblas_strsm(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    phi::dynload::cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static double *GEMM_ALLOC(ARGS... args) {
    return phi::dynload::cblas_dgemm_alloc(args...);
  }

  template <typename... ARGS>
  static void GEMM_PACK(ARGS... args) {
    phi::dynload::cblas_dgemm_pack(args...);
  }

  template <typename... ARGS>
  static void GEMM_COMPUTE(ARGS... args) {
    phi::dynload::cblas_dgemm_compute(args...);
  }

  template <typename... ARGS>
  static void GEMM_FREE(ARGS... args) {
    phi::dynload::cblas_dgemm_free(args...);
  }

#ifdef PADDLE_WITH_LIBXSMM
  template <typename... ARGS>
  static void SMM_GEMM(ARGS... args) {
    libxsmm_dgemm(args...);
  }
#endif

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    phi::dynload::cblas_dgemv(args...);
  }

  template <typename... ARGS>
  static double DOT(ARGS... args) {
    return detail::dot(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    phi::dynload::cblas_dgemm_batch(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    phi::dynload::cblas_dtrsm(args...);
  }
};

template <>
struct CBlas<phi::complex64> {
  template <typename... ARGS>
  static void AXPY(int64_t n,
                   const phi::complex64 alpha,
                   const phi::complex64 *X,
                   int64_t incX,
                   phi::complex64 *Y,
                   int64_t incY) {
    detail::axpy(n, alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans,
                   int M,
                   int N,
                   phi::complex64 alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *X,
                   int incx,
                   phi::complex64 beta,
                   phi::complex64 *Y,
                   int incy) {
    const void *a_ = (const void *)(A);
    const void *x_ = (const void *)(X);
    void *y_ = static_cast<void *>(Y);
    phi::dynload::cblas_cgemv(
        layout, trans, M, N, &alpha, a_, lda, x_, incx, &beta, y_, incy);
  }

  template <typename... ARGS>
  static void GEMM(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   int M,
                   int N,
                   int K,
                   phi::complex64 alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *B,
                   int ldb,
                   phi::complex64 beta,
                   phi::complex64 *C,
                   int ldc) {
    const void *a_ = (const void *)(A);
    const void *b_ = (const void *)(B);
    void *c_ = static_cast<void *>(C);
    phi::dynload::cblas_cgemm(layout,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              &alpha,
                              a_,
                              lda,
                              b_,
                              ldb,
                              &beta,
                              c_,
                              ldc);
  }

  static void TRSM(CBLAS_LAYOUT layout,
                   CBLAS_SIDE side,
                   CBLAS_UPLO uplo,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_DIAG diag,
                   int M,
                   int N,
                   phi::complex64 alpha,
                   const phi::complex64 *A,
                   int lda,
                   phi::complex64 *B,
                   int ldb) {
    const void *a_ = (const void *)(A);
    void *b_ = static_cast<void *>(B);
    phi::dynload::cblas_ctrsm(
        layout, side, uplo, trans_a, diag, M, N, &alpha, a_, lda, b_, ldb);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE *trans_a,
                         CBLAS_TRANSPOSE *trans_b,
                         int *M,
                         int *N,
                         int *K,
                         phi::complex64 *alpha,
                         const phi::complex64 **A,
                         const int *lda,
                         const phi::complex64 **B,
                         const int *ldb,
                         phi::complex64 *beta,
                         phi::complex64 **C,
                         const int *ldc,
                         int group_count,
                         int *group_size) {
    const void **A_void = (const void **)(&(*A));
    const void **B_void = (const void **)(&(*B));
    void **C_void = reinterpret_cast<void **>(C);

    phi::dynload::cblas_cgemm_batch(layout,
                                    trans_a,
                                    trans_b,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    A_void,
                                    lda,
                                    B_void,
                                    ldb,
                                    beta,
                                    C_void,
                                    ldc,
                                    group_count,
                                    group_size);
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    phi::dynload::cblas_cgemm_batch(args...);
  }
};

template <>
struct CBlas<phi::complex128> {
  template <typename... ARGS>
  static void AXPY(int64_t n,
                   const phi::complex128 alpha,
                   const phi::complex128 *X,
                   int64_t incX,
                   phi::complex128 *Y,
                   int64_t incY) {
    detail::axpy(n, alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans,
                   int M,
                   int N,
                   phi::complex128 alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *X,
                   int incx,
                   phi::complex128 beta,
                   phi::complex128 *Y,
                   int incy) {
    const void *a_ = (const void *)(A);
    const void *x_ = (const void *)(X);
    void *y_ = static_cast<void *>(Y);
    phi::dynload::cblas_zgemv(
        layout, trans, M, N, &alpha, a_, lda, x_, incx, &beta, y_, incy);
  }

  template <typename... ARGS>
  static void GEMM(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   int M,
                   int N,
                   int K,
                   phi::complex128 alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *B,
                   int ldb,
                   phi::complex128 beta,
                   phi::complex128 *C,
                   int ldc) {
    const void *a_ = (const void *)(A);
    const void *b_ = (const void *)(B);
    void *c_ = static_cast<void *>(C);
    phi::dynload::cblas_zgemm(layout,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              &alpha,
                              a_,
                              lda,
                              b_,
                              ldb,
                              &beta,
                              c_,
                              ldc);
  }

  static void TRSM(CBLAS_LAYOUT layout,
                   CBLAS_SIDE side,
                   CBLAS_UPLO uplo,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_DIAG diag,
                   int M,
                   int N,
                   phi::complex128 alpha,
                   const phi::complex128 *A,
                   int lda,
                   phi::complex128 *B,
                   int ldb) {
    const void *a_ = (const void *)(A);
    void *b_ = static_cast<void *>(B);
    phi::dynload::cblas_ztrsm(
        layout, side, uplo, trans_a, diag, M, N, &alpha, a_, lda, b_, ldb);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE *trans_a,
                         CBLAS_TRANSPOSE *trans_b,
                         int *M,
                         int *N,
                         int *K,
                         phi::complex128 *alpha,
                         const phi::complex128 **A,
                         const int *lda,
                         const phi::complex128 **B,
                         const int *ldb,
                         phi::complex128 *beta,
                         phi::complex128 **C,
                         const int *ldc,
                         int group_count,
                         int *group_size) {
    const void **A_void = (const void **)(&(*A));
    const void **B_void = (const void **)(&(*B));
    void **C_void = reinterpret_cast<void **>(C);

    phi::dynload::cblas_zgemm_batch(layout,
                                    trans_a,
                                    trans_b,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    A_void,
                                    lda,
                                    B_void,
                                    ldb,
                                    beta,
                                    C_void,
                                    ldc,
                                    group_count,
                                    group_size);
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    phi::dynload::cblas_zgemm_batch(args...);
  }
};

#elif defined(PADDLE_WITH_HML)
template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    phi::dynload::cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    phi::dynload::cblas_sgemv(args...);
  }

  template <typename... ARGS>
  static float DOT(ARGS... args) {
    return detail::dot(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    phi::dynload::cblas_strsm(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    phi::dynload::cblas_sgemm_batch(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    phi::dynload::cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    phi::dynload::cblas_dgemv(args...);
  }

  template <typename... ARGS>
  static double DOT(ARGS... args) {
    return detail::dot(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    phi::dynload::cblas_dgemm_batch(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    phi::dynload::cblas_dtrsm(args...);
  }
};

template <>
struct CBlas<phi::complex64> {
  template <typename... ARGS>
  static void AXPY(int64_t n,
                   const phi::complex64 alpha,
                   const phi::complex64 *X,
                   int64_t incX,
                   phi::complex64 *Y,
                   int64_t incY) {
    detail::axpy(n, alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans,
                   int M,
                   int N,
                   phi::complex64 alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *X,
                   int incx,
                   phi::complex64 beta,
                   phi::complex64 *Y,
                   int incy) {
    const void *a_ = (const void *)(A);
    const void *x_ = (const void *)(X);
    void *y_ = static_cast<void *>(Y);
    phi::dynload::cblas_cgemv(
        layout, trans, M, N, &alpha, a_, lda, x_, incx, &beta, y_, incy);
  }

  template <typename... ARGS>
  static void GEMM(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   int M,
                   int N,
                   int K,
                   phi::complex64 alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *B,
                   int ldb,
                   phi::complex64 beta,
                   phi::complex64 *C,
                   int ldc) {
    const void *a_ = (const void *)(A);
    const void *b_ = (const void *)(B);
    void *c_ = static_cast<void *>(C);
    phi::dynload::cblas_cgemm(layout,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              &alpha,
                              a_,
                              lda,
                              b_,
                              ldb,
                              &beta,
                              c_,
                              ldc);
  }

  static void TRSM(CBLAS_LAYOUT layout,
                   CBLAS_SIDE side,
                   CBLAS_UPLO uplo,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_DIAG diag,
                   int M,
                   int N,
                   phi::complex64 alpha,
                   const phi::complex64 *A,
                   int lda,
                   phi::complex64 *B,
                   int ldb) {
    const void *a_ = (const void *)(A);
    void *b_ = static_cast<void *>(B);
    phi::dynload::cblas_ctrsm(
        layout, side, uplo, trans_a, diag, M, N, &alpha, a_, lda, b_, ldb);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE *trans_a,
                         CBLAS_TRANSPOSE *trans_b,
                         int *M,
                         int *N,
                         int *K,
                         phi::complex64 *alpha,
                         const phi::complex64 **A,
                         const int *lda,
                         const phi::complex64 **B,
                         const int *ldb,
                         phi::complex64 *beta,
                         phi::complex64 **C,
                         const int *ldc,
                         int group_count,
                         int *group_size) {
    const void **A_void = (const void **)(&(*A));
    const void **B_void = (const void **)(&(*B));
    void **C_void = reinterpret_cast<void **>(C);

    phi::dynload::cblas_cgemm_batch(layout,
                                    trans_a,
                                    trans_b,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    A_void,
                                    lda,
                                    B_void,
                                    ldb,
                                    beta,
                                    C_void,
                                    ldc,
                                    group_count,
                                    group_size);
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    phi::dynload::cblas_cgemm_batch(args...);
  }
};

template <>
struct CBlas<phi::complex128> {
  template <typename... ARGS>
  static void AXPY(int64_t n,
                   const phi::complex128 alpha,
                   const phi::complex128 *X,
                   int64_t incX,
                   phi::complex128 *Y,
                   int64_t incY) {
    detail::axpy(n, alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans,
                   int M,
                   int N,
                   phi::complex128 alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *X,
                   int incx,
                   phi::complex128 beta,
                   phi::complex128 *Y,
                   int incy) {
    const void *a_ = (const void *)(A);
    const void *x_ = (const void *)(X);
    void *y_ = static_cast<void *>(Y);
    phi::dynload::cblas_zgemv(
        layout, trans, M, N, &alpha, a_, lda, x_, incx, &beta, y_, incy);
  }

  template <typename... ARGS>
  static void GEMM(CBLAS_LAYOUT layout,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_TRANSPOSE trans_b,
                   int M,
                   int N,
                   int K,
                   phi::complex128 alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *B,
                   int ldb,
                   phi::complex128 beta,
                   phi::complex128 *C,
                   int ldc) {
    const void *a_ = (const void *)(A);
    const void *b_ = (const void *)(B);
    void *c_ = static_cast<void *>(C);
    phi::dynload::cblas_zgemm(layout,
                              trans_a,
                              trans_b,
                              M,
                              N,
                              K,
                              &alpha,
                              a_,
                              lda,
                              b_,
                              ldb,
                              &beta,
                              c_,
                              ldc);
  }

  static void TRSM(CBLAS_LAYOUT layout,
                   CBLAS_SIDE side,
                   CBLAS_UPLO uplo,
                   CBLAS_TRANSPOSE trans_a,
                   CBLAS_DIAG diag,
                   int M,
                   int N,
                   phi::complex128 alpha,
                   const phi::complex128 *A,
                   int lda,
                   phi::complex128 *B,
                   int ldb) {
    const void *a_ = (const void *)(A);
    void *b_ = static_cast<void *>(B);
    phi::dynload::cblas_ztrsm(
        layout, side, uplo, trans_a, diag, M, N, &alpha, a_, lda, b_, ldb);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(CBLAS_LAYOUT layout,
                         CBLAS_TRANSPOSE *trans_a,
                         CBLAS_TRANSPOSE *trans_b,
                         int *M,
                         int *N,
                         int *K,
                         phi::complex128 *alpha,
                         const phi::complex128 **A,
                         const int *lda,
                         const phi::complex128 **B,
                         const int *ldb,
                         phi::complex128 *beta,
                         phi::complex128 **C,
                         const int *ldc,
                         int group_count,
                         int *group_size) {
    const void **A_void = (const void **)(&(*A));
    const void **B_void = (const void **)(&(*B));
    void **C_void = reinterpret_cast<void **>(C);

    phi::dynload::cblas_zgemm_batch(layout,
                                    trans_a,
                                    trans_b,
                                    M,
                                    N,
                                    K,
                                    alpha,
                                    A_void,
                                    lda,
                                    B_void,
                                    ldb,
                                    beta,
                                    C_void,
                                    ldc,
                                    group_count,
                                    group_size);
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    phi::dynload::cblas_zgemm_batch(args...);
  }
};

#else

template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_sgemv(args...);
  }

  template <typename... ARGS>
  static float DOT(ARGS... args) {
    return detail::dot(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    cblas_strsm(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_dgemv(args...);
  }

  template <typename... ARGS>
  static double DOT(ARGS... args) {
    return detail::dot(args...);
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    cblas_dtrsm(args...);
  }
};

template <>
struct CBlas<phi::complex64> {
  template <typename... ARGS>
  static void AXPY(int64_t n,
                   const phi::complex64 alpha,
                   const phi::complex64 *X,
                   int64_t incX,
                   phi::complex64 *Y,
                   int64_t incY) {
    detail::axpy(n, alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const int M,
                   const int N,
                   const phi::complex64 alpha,
                   const phi::complex64 *A,
                   const int lda,
                   const phi::complex64 *X,
                   const int incX,
                   const phi::complex64 beta,
                   phi::complex64 *Y,
                   const int incY) {
    cblas_cgemv(layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
  }

  template <typename... ARGS>
  static void GEMM(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB,
                   const int M,
                   const int N,
                   const int K,
                   const phi::complex64 alpha,
                   const phi::complex64 *A,
                   const int lda,
                   const phi::complex64 *B,
                   const int ldb,
                   const phi::complex64 beta,
                   phi::complex64 *C,
                   const int ldc) {
    cblas_cgemm(
        layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  static void TRSM(const CBLAS_LAYOUT layout,
                   const CBLAS_SIDE side,
                   const CBLAS_UPLO uplo,
                   const CBLAS_TRANSPOSE transA,
                   const CBLAS_DIAG diag,
                   const int M,
                   const int N,
                   const phi::complex64 alpha,
                   const phi::complex64 *A,
                   const int lda,
                   phi::complex64 *B,
                   const int ldb) {
    cblas_ctrsm(layout, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb);
  }
};

template <>
struct CBlas<phi::complex128> {
  template <typename... ARGS>
  static void AXPY(int64_t n,
                   const phi::complex128 alpha,
                   const phi::complex128 *X,
                   int64_t incX,
                   phi::complex128 *Y,
                   int64_t incY) {
    detail::axpy(n, alpha, X, incX, Y, incY);
  }

  template <typename... ARGS>
  static void GEMV(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const int M,
                   const int N,
                   const phi::complex128 alpha,
                   const phi::complex128 *A,
                   const int lda,
                   const phi::complex128 *X,
                   const int incX,
                   const phi::complex128 beta,
                   phi::complex128 *Y,
                   const int incY) {
    cblas_zgemv(layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
  }

  template <typename... ARGS>
  static void GEMM(const CBLAS_LAYOUT layout,
                   const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB,
                   const int M,
                   const int N,
                   const int K,
                   const phi::complex128 alpha,
                   const phi::complex128 *A,
                   const int lda,
                   const phi::complex128 *B,
                   const int ldb,
                   const phi::complex128 beta,
                   phi::complex128 *C,
                   const int ldc) {
    cblas_zgemm(
        layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }

  static void TRSM(const CBLAS_LAYOUT layout,
                   const CBLAS_SIDE side,
                   const CBLAS_UPLO uplo,
                   const CBLAS_TRANSPOSE transA,
                   const CBLAS_DIAG diag,
                   const int M,
                   const int N,
                   const phi::complex128 alpha,
                   const phi::complex128 *A,
                   const int lda,
                   phi::complex128 *B,
                   const int ldb) {
    cblas_ztrsm(layout, side, uplo, transA, diag, M, N, &alpha, A, lda, B, ldb);
  }
};

#endif

template <>
struct CBlas<phi::float16> {
  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    detail::axpy(args...);
  }

  template <typename... ARGS>
  static phi::float16 DOT(ARGS... args) {
    return detail::dot(args...);
  }

  static void GEMM(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 GEMM not supported on CPU, please check your code"));
  }

  static void SMM_GEMM(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 SMM_GEMM not supported on CPU, please check your code"));
  }
#ifdef PADDLE_WITH_MKLML
  static void GEMM_BATCH(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 GEMM_BATCH not supported on CPU, please check your code"));
  }
#endif
#ifdef PADDLE_WITH_HML
  static void GEMM_BATCH(...) {
    PADDLE_THROW(common::errors::Unimplemented(
        "float16 GEMM_BATCH not supported on CPU, please check your code"));
  }
#endif
};

#ifdef PADDLE_WITH_MKLML
template <>
template <typename T>
T *Blas<CPUContext>::GEMM_ALLOC(const CBLAS_IDENTIFIER id,
                                const int M,
                                const int N,
                                const int K) const {
  return CBlas<T>::GEMM_ALLOC(id, M, N, K);
}

template <>
template <typename T>
void Blas<CPUContext>::GEMM_PACK(const CBLAS_IDENTIFIER id,
                                 const CBLAS_TRANSPOSE trans,
                                 int M,
                                 int N,
                                 int K,
                                 const T alpha,
                                 const T *src,
                                 const int ld,
                                 T *dst) const {
  CBlas<T>::GEMM_PACK(CblasRowMajor, id, trans, M, N, K, alpha, src, ld, dst);
}

template <>
template <typename T>
void Blas<CPUContext>::GEMM_COMPUTE(int transA,
                                    int transB,
                                    int M,
                                    int N,
                                    int K,
                                    const T *A,
                                    const int lda,
                                    const T *B,
                                    const int ldb,
                                    T beta,
                                    T *C,
                                    const int ldc) const {
  CBlas<T>::GEMM_COMPUTE(
      CblasRowMajor, transA, transB, M, N, K, A, lda, B, ldb, beta, C, ldc);
}

template <>
template <typename T>
void Blas<CPUContext>::GEMM_FREE(T *data) const {
  CBlas<T>::GEMM_FREE(data);
}
#endif

template <>
template <typename T>
void Blas<CPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB,
                            int64_t M,
                            int64_t N,
                            int64_t K,
                            T alpha,
                            const T *A,
                            const T *B,
                            T beta,
                            T *C) const {
  if (M > std::numeric_limits<int>::max() ||
      N > std::numeric_limits<int>::max() ||
      K > std::numeric_limits<int>::max()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "CPU GEMM only supports M, N and K not larger than INT_MAX. "
        "Expected M <= %d, N <= %d and K <= %d, but received M = %ld, "
        "N = %ld, K = %ld.",
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        M,
        N,
        K));
  }
  int lda = static_cast<int>((transA == CblasNoTrans) ? K : M);
  int ldb = static_cast<int>((transB == CblasNoTrans) ? N : K);
  int ldc = static_cast<int>(N);
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 static_cast<int>(M),
                 static_cast<int>(N),
                 static_cast<int>(K),
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T, typename U>
void Blas<CPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB,
                            int64_t M,
                            int64_t N,
                            int64_t K,
                            U alpha,
                            const T *A,
                            const T *B,
                            U beta,
                            T *C) const {
  if (M > std::numeric_limits<int>::max() ||
      N > std::numeric_limits<int>::max() ||
      K > std::numeric_limits<int>::max()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "CPU GEMM only supports M, N and K not larger than INT_MAX. "
        "Expected M <= %d, N <= %d and K <= %d, but received M = %ld, "
        "N = %ld, K = %ld.",
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        M,
        N,
        K));
  }
  int lda = static_cast<int>((transA == CblasNoTrans) ? K : M);
  int ldb = static_cast<int>((transB == CblasNoTrans) ? N : K);
  int ldc = static_cast<int>(N);
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 static_cast<int>(M),
                 static_cast<int>(N),
                 static_cast<int>(K),
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T>
void Blas<CPUContext>::GEMM(bool transA,
                            bool transB,
                            int M,
                            int N,
                            int K,
                            T alpha,
                            const T *A,
                            int lda,
                            const T *B,
                            int ldb,
                            T beta,
                            T *C,
                            int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor,
                 transA == false ? CblasNoTrans : CblasTrans,
                 transB == false ? CblasNoTrans : CblasTrans,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <>
template <typename T>
void Blas<CPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                            CBLAS_TRANSPOSE transB,
                            int M,
                            int N,
                            int K,
                            T alpha,
                            const T *A,
                            int lda,
                            const T *B,
                            int ldb,
                            T beta,
                            T *C,
                            int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor,
                 transA,
                 transB,
                 M,
                 N,
                 K,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const DenseTensor &mat_a,
                                 bool trans_a,
                                 const DenseTensor &mat_b,
                                 bool trans_b,
                                 T alpha,
                                 DenseTensor *mat_out,
                                 T beta) const {
  const auto &dim_a = mat_a.dims();
  const auto &dim_b = mat_b.dims();
  const auto &dim_out = mat_out->dims();
  PADDLE_ENFORCE_EQ(
      dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
      true,
      common::errors::InvalidArgument(
          "The input and output of matmul should be matrix, the dim size must "
          "be 2,"
          "but received dim size input_a:%d, input_b:%d, output:%d",
          dim_a.size(),
          dim_b.size(),
          dim_out.size()));
  PADDLE_ENFORCE_EQ(
      mat_a.place() == mat_b.place() && mat_a.place() == mat_out->place(),
      true,
      common::errors::InvalidArgument("The places of matrices in the matmul "
                                      "should be same, please check your "
                                      "code."));

  const int64_t K_64 = !trans_a ? dim_a[1] : dim_a[0];
  PADDLE_ENFORCE_LE_INT_MAX(dim_out[0], "dim_out[0]");
  PADDLE_ENFORCE_LE_INT_MAX(dim_out[1], "dim_out[1]");
  PADDLE_ENFORCE_LE_INT_MAX(K_64, "cblas GEMM K");
  int M = static_cast<int>(dim_out[0]);
  int N = static_cast<int>(dim_out[1]);
  int K = static_cast<int>(K_64);

  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !trans_b ? CblasNoTrans : CblasTrans;

  this->GEMM(transA,
             transB,
             M,
             N,
             K,
             alpha,
             mat_a.data<T>(),
             mat_b.data<T>(),
             beta,
             mat_out->data<T>());
}

template <>
template <typename T>
void Blas<CPUContext>::AXPY(int64_t n, T alpha, const T *x, T *y) const {
  CBlas<T>::AXPY(n, alpha, x, 1, y, 1);
}

template <>
template <typename T>
T Blas<CPUContext>::DOT(
    int64_t n, const T *x, int64_t incx, const T *y, int64_t incy) const {
  return detail::dot(n, x, incx, y, incy);
}

template <>
template <typename T>
void Blas<CPUContext>::GEMV(bool trans_a,
                            int M,
                            int N,
                            T alpha,
                            const T *A,
                            const T *B,
                            T beta,
                            T *C) const {
  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBlas<T>::GEMV(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
template <typename T>
void Blas<CPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                   CBLAS_TRANSPOSE transB,
                                   int64_t M,
                                   int64_t N,
                                   int64_t K,
                                   T alpha,
                                   const T *A,
                                   const T *B,
                                   T beta,
                                   T *C,
                                   int64_t batchCount,
                                   int64_t strideA,
                                   int64_t strideB) const {
  PADDLE_ENFORCE_NOT_NULL(
      A, common::errors::InvalidArgument("Pointer A should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      B, common::errors::InvalidArgument("Pointer B should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      C, common::errors::InvalidArgument("Pointer C should not be null."));

  if (M > std::numeric_limits<int>::max() ||
      N > std::numeric_limits<int>::max() ||
      K > std::numeric_limits<int>::max() ||
      batchCount > std::numeric_limits<int>::max()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "CPU BatchedGEMM only supports M, N, K and batchCount not larger "
        "than INT_MAX. Expected M <= %d, N <= %d, K <= %d and "
        "batchCount <= %d, but received M = %ld, N = %ld, K = %ld, "
        "batchCount = %ld.",
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        M,
        N,
        K,
        batchCount));
  }

#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
  int M_int = static_cast<int>(M);
  int N_int = static_cast<int>(N);
  int K_int = static_cast<int>(K);
  int batch_count_int = static_cast<int>(batchCount);
  int lda = (transA == CblasNoTrans) ? K_int : M_int;
  int ldb = (transB == CblasNoTrans) ? N_int : K_int;
  int ldc = N_int;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }
  CBlas<T>::GEMM_BATCH(CblasRowMajor,
                       &transA,
                       &transB,
                       &M_int,
                       &N_int,
                       &K_int,
                       &alpha,
                       a_array.data(),
                       &lda,
                       b_array.data(),
                       &ldb,
                       &beta,
                       c_array.data(),
                       &ldc,
                       1 /* group_count */,
                       &batch_count_int);
#else
  for (int64_t k = 0; k < batchCount; ++k) {
    auto *Ak = &A[k * strideA];
    auto *Bk = &B[k * strideB];
    auto *Ck = &C[k * M * N];
    this->template GEMM<T>(transA,
                           transB,
                           static_cast<int>(M),
                           static_cast<int>(N),
                           static_cast<int>(K),
                           alpha,
                           Ak,
                           Bk,
                           beta,
                           Ck);
  }
#endif
}

template <>
template <typename T, typename U>
void Blas<CPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                   CBLAS_TRANSPOSE transB,
                                   int64_t M,
                                   int64_t N,
                                   int64_t K,
                                   U alpha,
                                   const T *A,
                                   const T *B,
                                   U beta,
                                   T *C,
                                   int64_t batchCount,
                                   int64_t strideA,
                                   int64_t strideB) const {
  PADDLE_ENFORCE_NOT_NULL(
      A, common::errors::InvalidArgument("Pointer A should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      B, common::errors::InvalidArgument("Pointer B should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      C, common::errors::InvalidArgument("Pointer C should not be null."));
  if (M > std::numeric_limits<int>::max() ||
      N > std::numeric_limits<int>::max() ||
      K > std::numeric_limits<int>::max() ||
      batchCount > std::numeric_limits<int>::max()) {
    PADDLE_THROW(common::errors::Unimplemented(
        "CPU BatchedGEMM does not support M, N, K or batchCount larger than "
        "INT_MAX."));
  }

#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
  int M_int = static_cast<int>(M);
  int N_int = static_cast<int>(N);
  int K_int = static_cast<int>(K);
  int batch_count_int = static_cast<int>(batchCount);
  int lda = (transA == CblasNoTrans) ? K_int : M_int;
  int ldb = (transB == CblasNoTrans) ? N_int : K_int;
  int ldc = N_int;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }

  CBlas<T>::GEMM_BATCH(CblasRowMajor,
                       &transA,
                       &transB,
                       &M_int,
                       &N_int,
                       &K_int,
                       &alpha,
                       a_array.data(),
                       &lda,
                       b_array.data(),
                       &ldb,
                       &beta,
                       c_array.data(),
                       &ldc,
                       1 /* group_count */,
                       &batch_count_int);
#else
  for (int64_t k = 0; k < batchCount; ++k) {
    auto *Ak = &A[k * strideA];
    auto *Bk = &B[k * strideB];
    auto *Ck = &C[k * M * N];
    this->template GEMM<T>(transA,
                           transB,
                           static_cast<int>(M),
                           static_cast<int>(N),
                           static_cast<int>(K),
                           alpha,
                           Ak,
                           Bk,
                           beta,
                           Ck);
  }
#endif
}

template <>
template <typename T>
void Blas<CPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                   CBLAS_TRANSPOSE transB,
                                   int M,
                                   int N,
                                   int K,
                                   T alpha,
                                   const T **A,
                                   const T **B,
                                   T beta,
                                   T **C,
                                   int batchCount) const {
#if defined(PADDLE_WITH_MKLML) || defined(PADDLE_WITH_HML)
  const int lda = (std::max)((transA == CblasNoTrans) ? K : M, 1);
  const int ldb = (std::max)((transB == CblasNoTrans) ? N : K, 1);
  const int ldc = (std::max)(N, 1);
  CBlas<T>::GEMM_BATCH(CblasRowMajor,
                       &transA,
                       &transB,
                       &M,
                       &N,
                       &K,
                       &alpha,
                       A,
                       &lda,
                       B,
                       &ldb,
                       &beta,
                       C,
                       &ldc,
                       1 /* group_count */,
                       &batchCount);
#else
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
#endif
}

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)  // @{ Group Blas MKLML: BatchedGEMMWithHead
template <>
template <typename T>
void Blas<CPUContext>::BatchedGEMMWithHead(CBLAS_TRANSPOSE transA,
                                           CBLAS_TRANSPOSE transB,
                                           int W1,
                                           int H1,
                                           int W2,
                                           int H2,
                                           T alpha,
                                           const T *A,
                                           const T *B,
                                           T beta,
                                           T *C,
                                           int batchCount,
                                           int64_t strideA,
                                           int64_t strideB,
                                           int64_t head_number,
                                           bool split_b_vertical) const {
  int lda = (transA == CblasNoTrans) ? W1 : H1;
  int ldb = (transB == CblasNoTrans) ? W2 : H2;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);

  if (split_b_vertical) {
    int ldc = W2;
    int sub_width = W2 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W2 / head_number)
                                : i * (W2 / head_number) * H2;
      int sub_matC_offset = i * W2 / head_number;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor,
                           &transA,
                           &transB,
                           &H1,
                           &sub_width,
                           &H2,
                           &alpha,
                           a_array.data(),
                           &lda,
                           b_array.data(),
                           &ldb,
                           &beta,
                           c_array.data(),
                           &ldc,
                           1 /* group_count */,
                           &batchCount);
    }

  } else {
    PADDLE_ENFORCE_EQ(
        W1,
        H2,
        common::errors::InvalidArgument(
            "The first matrix width should be same as second matrix height,"
            "but received first matrix width %d"
            ", second matrix height %d",
            W1,
            H2));
    int ldc = W2 * head_number;
    int sub_width = W1 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W1 / head_number) * W2
                                : i * (W1 / head_number);
      int sub_matC_offset = i * W2;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * head_number * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor,
                           &transA,
                           &transB,
                           &H1,
                           &W2,
                           &sub_width,
                           &alpha,
                           a_array.data(),
                           &lda,
                           b_array.data(),
                           &ldb,
                           &beta,
                           c_array.data(),
                           &ldc,
                           1 /* group_count */,
                           &batchCount);
    }
  }
}
#endif  // @} End Group Blas MKLML: BatchedGEMMWithHead

#if defined(PADDLE_WITH_HML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)  // @{ Group Blas HML: BatchedGEMMWithHead
template <>
template <typename T>
void Blas<CPUContext>::BatchedGEMMWithHead(CBLAS_TRANSPOSE transA,
                                           CBLAS_TRANSPOSE transB,
                                           int W1,
                                           int H1,
                                           int W2,
                                           int H2,
                                           T alpha,
                                           const T *A,
                                           const T *B,
                                           T beta,
                                           T *C,
                                           int batchCount,
                                           int64_t strideA,
                                           int64_t strideB,
                                           int64_t head_number,
                                           bool split_b_vertical) const {
  int lda = (transA == CblasNoTrans) ? W1 : H1;
  int ldb = (transB == CblasNoTrans) ? W2 : H2;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);

  if (split_b_vertical) {
    int ldc = W2;
    int sub_width = W2 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W2 / head_number)
                                : i * (W2 / head_number) * H2;
      int sub_matC_offset = i * W2 / head_number;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor,
                           &transA,
                           &transB,
                           &H1,
                           &sub_width,
                           &H2,
                           &alpha,
                           a_array.data(),
                           &lda,
                           b_array.data(),
                           &ldb,
                           &beta,
                           c_array.data(),
                           &ldc,
                           1 /* group_count */,
                           &batchCount);
    }

  } else {
    PADDLE_ENFORCE_EQ(
        W1,
        H2,
        common::errors::InvalidArgument(
            "The first matrix width should be same as second matrix height,"
            "but received first matrix width %d"
            ", second matrix height %d",
            W1,
            H2));
    int ldc = W2 * head_number;
    int sub_width = W1 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W1 / head_number) * W2
                                : i * (W1 / head_number);
      int sub_matC_offset = i * W2;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * head_number * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor,
                           &transA,
                           &transB,
                           &H1,
                           &W2,
                           &sub_width,
                           &alpha,
                           a_array.data(),
                           &lda,
                           b_array.data(),
                           &ldb,
                           &beta,
                           c_array.data(),
                           &ldc,
                           1 /* group_count */,
                           &batchCount);
    }
  }
}
#endif  // @{ Group Blas HML: BatchedGEMMWithHead

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(
    const int M, const int N, const int K, const T *A, const T *B, T *C) const {
  this->template GEMM<T>(CblasRowMajor,
                         CblasNoTrans,
                         CblasNoTrans,
                         M,
                         N,
                         K,
                         static_cast<T>(1),
                         A,
                         K,
                         B,
                         N,
                         static_cast<T>(0),
                         C,
                         N);
}

template <>
template <typename T>
void Blas<CPUContext>::MatMul(
    const int M, const int N, const int K, const T *A, const T *B, T *C) const {
#ifdef PADDLE_WITH_LIBXSMM
  // Refer to https://github.com/hfp/libxsmm/blob/master/README.md
  // But the threshold is custom constexpr int LIBXSMM_THRESHOLD = 20 * 20 * 20;

  // Since the matrix is very small,
  // so the unit of calculation is already very fast,
  // and the if( M*N*K < LIBXSMM_THRESHOLD) would be overhead,
  // use xsmm directly.
  // Note: SMM use ColMajor
  const char transa = 'N';
  const char transb = 'N';
  const T alpha = static_cast<T>(1);
  const T beta = static_cast<T>(0);
  CBlas<T>::SMM_GEMM(
      &transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
  return;
#endif

  CBlas<T>::GEMM(CblasRowMajor,
                 CblasNoTrans,
                 CblasNoTrans,
                 M,
                 N,
                 K,
                 static_cast<T>(1),
                 A,
                 K,
                 B,
                 N,
                 static_cast<T>(0),
                 C,
                 N);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const DenseTensor &mat_a,
                                 const MatDescriptor &dim_a,
                                 const DenseTensor &mat_b,
                                 const MatDescriptor &dim_b,
                                 T alpha,
                                 DenseTensor *mat_out,
                                 T beta) const {
  MatMul(mat_a.data<T>(),
         dim_a,
         mat_b.data<T>(),
         dim_b,
         alpha,
         mat_out->data<T>(),
         beta);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const T *mat_a,
                                 const MatDescriptor &dim_a,
                                 const T *mat_b,
                                 const MatDescriptor &dim_b,
                                 T alpha,
                                 T *mat_out,
                                 T beta) const {
  PADDLE_ENFORCE_EQ(
      dim_a.width_,
      dim_b.height_,
      common::errors::InvalidArgument(
          "The first matrix width should be same as second matrix height,"
          "but received first matrix width %d"
          ", second matrix height %d",
          dim_a.width_,
          dim_b.height_));

  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;
  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    this->template GEMM<T>(transA,
                           transB,
                           dim_a.height_,
                           dim_b.width_,
                           dim_a.width_,
                           alpha,
                           mat_a,
                           mat_b,
                           beta,
                           mat_out);
  } else {
    PADDLE_ENFORCE_EQ(
        dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
            dim_b.batch_size_ == 0,
        true,
        common::errors::InvalidArgument(
            "dim_a.batch_size should be equal to dim_b.batch_size, or "
            "one of dim_a.batch_size and dim_b.batch_size should be 0. "
            "But got dim_a.batch_size = %d, dim_b.batch_size = %d.",
            dim_a.batch_size_,
            dim_b.batch_size_));
    this->template BatchedGEMM<T>(
        transA,
        transB,
        dim_a.height_,
        dim_b.width_,
        dim_a.width_,
        alpha,
        mat_a,
        mat_b,
        beta,
        mat_out,
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_);
  }
}

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
// @{ Group Blas MKLML: MatMulWithHead
/*
 * Multiple two matrixes with multiple heads
 *
 * A new parameter, i.e head_number is added compared to normal MatMul.
 * The head_number describes the number of heads a matrix is vertically
 * split.
 *
 * When user calls this API, the multiplication of two big matrixes is split
 * into multiplication of several (head_number_) small matrixes. e.g. if Mat A
 * is [3, 24] and Mat B is [24, 4], when multiple A and B with head_number as
 * 4, Mat A will be split as 4 matrix of [3, 6] and Mat B will be
 * (horizontally) split as 4 matrix of [6, 4]. The result of final matrix
 * will be 4 matrix of [3, 4], i.e. [3, 16].
 * Another example is A is [3, 8], B is [2, 16], head_number is 4. In this
 * case, A will be split as [3, 2], B will be (vertically) split as
 * [2, 4]. The final result will be 4 matrix of 4 matrix of [3,4], i.e. [3, 16]
 */
template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMulWithHead(const DenseTensor &mat_a,
                                         const MatDescriptor &dim_a,
                                         const DenseTensor &mat_b,
                                         const MatDescriptor &dim_b,
                                         T alpha,
                                         int head_number,
                                         DenseTensor *mat_out,
                                         T beta,
                                         bool mat_b_split_vertical) const {
  PADDLE_ENFORCE_EQ(
      dim_a.width_ % head_number,
      0,
      common::errors::InvalidArgument(
          "The first input width must be some times the head number, "
          "but received first input width %d"
          ",  head_number %d",
          dim_a.width_,
          head_number));
  PADDLE_ENFORCE_GE(head_number,
                    1,
                    common::errors::InvalidArgument(
                        "The head number should be greater equal 1,"
                        "but received head number %d",
                        head_number));
  PADDLE_ENFORCE_LE(
      head_number,
      dim_a.width_,
      common::errors::InvalidArgument(
          "The head number should be less equal first input width,"
          "but received first input width %d"
          ",  head_number %d",
          dim_a.width_,
          head_number));
  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;

  if (mat_b_split_vertical) {
    PADDLE_ENFORCE_EQ(
        dim_b.height_,
        dim_a.width_ / head_number,
        common::errors::InvalidArgument(
            "The second input height should be equal than first input width,"
            "but received second input height %d, first input width %d",
            dim_b.height_,
            dim_a.width_ / head_number));
    PADDLE_ENFORCE_EQ(
        dim_a.width_ % head_number,
        0,
        common::errors::InvalidArgument(
            "The second input width should be some times the head number, "
            "but received second input width %d"
            ",  head_number %d",
            dim_b.width_,
            head_number));
  }

  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    int lda = !dim_a.trans_ ? dim_a.width_ : dim_a.height_;
    int ldb = !dim_b.trans_ ? dim_b.width_ : dim_b.height_;
    int sub_matA_offset;
    int sub_matB_offset;
    int sub_matC_offset;
    int sub_mat_M = dim_a.height_;
    int sub_mat_N;
    int sub_mat_K;
    int ldc;

    for (int i = 0; i < head_number; i++) {
      sub_matA_offset = dim_a.trans_
                            ? i * (dim_a.width_ / head_number) * dim_a.height_
                            : i * (dim_a.width_ / head_number);
      if (mat_b_split_vertical) {
        sub_matB_offset = dim_b.trans_
                              ? i * (dim_b.width_ / head_number) * dim_b.height_
                              : i * (dim_b.width_ / head_number);
        sub_matC_offset = i * dim_b.width_ / head_number;

        sub_mat_N = dim_b.width_ / head_number;
        sub_mat_K = dim_b.height_;

        ldc = dim_b.width_;
      } else {
        sub_matB_offset =
            dim_b.trans_ ? i * (dim_b.height_ / head_number)
                         : i * (dim_b.height_ / head_number) * dim_b.width_;
        sub_matC_offset = i * dim_b.width_;

        sub_mat_N = dim_b.width_;
        sub_mat_K = dim_a.width_ / head_number;

        ldc = head_number * dim_b.width_;
      }

      this->template GEMM<T>(transA,
                             transB,
                             sub_mat_M,
                             sub_mat_N,
                             sub_mat_K,
                             alpha,
                             mat_a.data<T>() + sub_matA_offset,
                             lda,
                             mat_b.data<T>() + sub_matB_offset,
                             ldb,
                             beta,
                             mat_out->data<T>() + sub_matC_offset,
                             ldc);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        (dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
         dim_b.batch_size_ == 0),
        true,
        common::errors::InvalidArgument(
            "The first input batch size should be equal than second input,"
            "either two input batch size is 0, but received first input batch "
            "size"
            " %d, second input batch size %d",
            dim_a.batch_size_,
            dim_b.batch_size_));

    this->template BatchedGEMMWithHead<T>(
        transA,
        transB,
        dim_a.width_,
        dim_a.height_,
        dim_b.width_,
        dim_b.height_,
        alpha,
        mat_a.data<T>(),
        mat_b.data<T>(),
        beta,
        mat_out->data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_,
        head_number,
        mat_b_split_vertical);
  }
}
#endif  // @} End Group Blas MKLML: MatMulWithHead

#if defined(PADDLE_WITH_HML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
// @{ Group Blas HML: MatMulWithHead
/*
 * Multiple two matrixes with multiple heads
 *
 * A new parameter, i.e head_number is added compared to normal MatMul.
 * The head_number describes the number of heads a matrix is vertically
 * split.
 *
 * When user calls this API, the multiplication of two big matrixes is split
 * into multiplication of several (head_number_) small matrixes. e.g. if Mat A
 * is [3, 24] and Mat B is [24, 4], when multiple A and B with head_number as
 * 4, Mat A will be split as 4 matrix of [3, 6] and Mat B will be
 * (horizontally) split as 4 matrix of [6, 4]. The result of final matrix
 * will be 4 matrix of [3, 4], i.e. [3, 16].
 * Another example is A is [3, 8], B is [2, 16], head_number is 4. In this
 * case, A will be split as [3, 2], B will be (vertically) split as
 * [2, 4]. The final result will be 4 matrix of 4 matrix of [3,4], i.e. [3, 16]
 */
template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMulWithHead(const DenseTensor &mat_a,
                                         const MatDescriptor &dim_a,
                                         const DenseTensor &mat_b,
                                         const MatDescriptor &dim_b,
                                         T alpha,
                                         int head_number,
                                         DenseTensor *mat_out,
                                         T beta,
                                         bool mat_b_split_vertical) const {
  PADDLE_ENFORCE_EQ(
      dim_a.width_ % head_number,
      0,
      common::errors::InvalidArgument(
          "The first input width must be some times the head number, "
          "but received first input width %d"
          ",  head_number %d",
          dim_a.width_,
          head_number));
  PADDLE_ENFORCE_GE(head_number,
                    1,
                    common::errors::InvalidArgument(
                        "The head number should be greater equal 1,"
                        "but received head number %d",
                        head_number));
  PADDLE_ENFORCE_LE(
      head_number,
      dim_a.width_,
      common::errors::InvalidArgument(
          "The head number should be less equal first input width,"
          "but received first input width %d"
          ",  head_number %d",
          dim_a.width_,
          head_number));
  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;

  if (mat_b_split_vertical) {
    PADDLE_ENFORCE_EQ(
        dim_b.height_,
        dim_a.width_ / head_number,
        common::errors::InvalidArgument(
            "The second input height should be equal than first input width,"
            "but received second input height %d, first input width %d",
            dim_b.height_,
            dim_a.width_ / head_number));
    PADDLE_ENFORCE_EQ(
        dim_a.width_ % head_number,
        0,
        common::errors::InvalidArgument(
            "The second input width should be some times the head number, "
            "but received second input width %d"
            ",  head_number %d",
            dim_b.width_,
            head_number));
  }

  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    int lda = !dim_a.trans_ ? dim_a.width_ : dim_a.height_;
    int ldb = !dim_b.trans_ ? dim_b.width_ : dim_b.height_;
    int sub_matA_offset;
    int sub_matB_offset;
    int sub_matC_offset;
    int sub_mat_M = dim_a.height_;
    int sub_mat_N;
    int sub_mat_K;
    int ldc;

    for (int i = 0; i < head_number; i++) {
      sub_matA_offset = dim_a.trans_
                            ? i * (dim_a.width_ / head_number) * dim_a.height_
                            : i * (dim_a.width_ / head_number);
      if (mat_b_split_vertical) {
        sub_matB_offset = dim_b.trans_
                              ? i * (dim_b.width_ / head_number) * dim_b.height_
                              : i * (dim_b.width_ / head_number);
        sub_matC_offset = i * dim_b.width_ / head_number;

        sub_mat_N = dim_b.width_ / head_number;
        sub_mat_K = dim_b.height_;

        ldc = dim_b.width_;
      } else {
        sub_matB_offset =
            dim_b.trans_ ? i * (dim_b.height_ / head_number)
                         : i * (dim_b.height_ / head_number) * dim_b.width_;
        sub_matC_offset = i * dim_b.width_;

        sub_mat_N = dim_b.width_;
        sub_mat_K = dim_a.width_ / head_number;

        ldc = head_number * dim_b.width_;
      }

      this->template GEMM<T>(transA,
                             transB,
                             sub_mat_M,
                             sub_mat_N,
                             sub_mat_K,
                             alpha,
                             mat_a.data<T>() + sub_matA_offset,
                             lda,
                             mat_b.data<T>() + sub_matB_offset,
                             ldb,
                             beta,
                             mat_out->data<T>() + sub_matC_offset,
                             ldc);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        (dim_a.batch_size_ == dim_b.batch_size_ || dim_a.batch_size_ == 0 ||
         dim_b.batch_size_ == 0),
        true,
        common::errors::InvalidArgument(
            "The first input batch size should be equal to second input,"
            "either two input batch size is 0, but received first input batch "
            "size"
            " %d, second input batch size %d",
            dim_a.batch_size_,
            dim_b.batch_size_));

    this->template BatchedGEMMWithHead<T>(
        transA,
        transB,
        dim_a.width_,
        dim_a.height_,
        dim_b.width_,
        dim_b.height_,
        alpha,
        mat_a.data<T>(),
        mat_b.data<T>(),
        beta,
        mat_out->data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_,
        dim_b.stride_,
        head_number,
        mat_b_split_vertical);
  }
}
#endif  // @} End Group Blas HML: MatMulWithHead

template <>
template <typename T>
void Blas<CPUContext>::TRSM(CBLAS_SIDE side,
                            CBLAS_UPLO uplo,
                            CBLAS_TRANSPOSE transA,
                            CBLAS_DIAG diag,
                            int64_t M,
                            int64_t N,
                            T alpha,
                            const T *A,
                            int64_t lda,
                            T *B,
                            int64_t ldb) const {
  const int m = detail::to_blas_int(M, "TRSM M");
  const int n = detail::to_blas_int(N, "TRSM N");
  const int lda_int = detail::to_blas_int(lda, "TRSM lda");
  const int ldb_int = detail::to_blas_int(ldb, "TRSM ldb");
  CBlas<T>::TRSM(CblasRowMajor,
                 side,
                 uplo,
                 transA,
                 diag,
                 m,
                 n,
                 alpha,
                 A,
                 lda_int,
                 B,
                 ldb_int);
}

}  // namespace funcs
}  // namespace phi
