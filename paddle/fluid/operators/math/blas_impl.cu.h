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

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/hipblas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasSaxpy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasSgemmStridedBatched(args...));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasDaxpy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::hipblasDgemmStridedBatched(args...));
  }
};

#if 0
template <>
struct CUBlas<platform::float16> {
  using float16 = platform::float16;

  static void GEMM(hipblasHandle_t handle, hipblasOperation_t transa,
                   hipblasOperation_t transb, int m, int n, int k,
                   const float16 *alpha, const float16 *A, int lda,
                   const float16 *B, int ldb, const float16 *beta, float16 *C,
                   int ldc) {
    PADDLE_ENFORCE(
        platform::dynload::hipblasHgemm(handle, transa, transb, m, n, k,
                                       reinterpret_cast<const hipblasHalf *>(alpha),
                                       reinterpret_cast<const hipblasHalf *>(A), lda,
                                       reinterpret_cast<const hipblasHalf *>(B), ldb,
                                       reinterpret_cast<const hipblasHalf *>(beta),
                                       reinterpret_cast<const hipblasHalf *>(C), ldc));
  }

  static void GEMM_BATCH(hipblasHandle_t handle, hipblasOperation_t transa,
                         hipblasOperation_t transb, int m, int n, int k,
                         const float16 *alpha, const float16 *A, int lda,
                         long long int strideA, const float16 *B,  // NOLINT
                         int ldb, long long int strideB,           // NOLINT
                         const float16 *beta, float16 *C, int ldc,
                         long long int strideC,  // NOLINT
                         int batchCount) {
#if 0
    PADDLE_ENFORCE(platform::dynload::hipblasHgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const __half *>(alpha),
        reinterpret_cast<const __half *>(A), lda, strideA,
        reinterpret_cast<const __half *>(B), ldb, strideB,
        reinterpret_cast<const __half *>(beta), reinterpret_cast<__half *>(C),
        ldc, strideC, batchCount));
#else
    PADDLE_THROW("HgemmStridedBatched is not supported on cuda <= 7.5");
#endif
  }
};
#endif

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                             CBLAS_TRANSPOSE transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             const T *B, T beta, T *C) const {
  // Note that hipblas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

  CUBlas<T>::GEMM(context_.hipblas_handle(), cuTransB, cuTransA, N, M, K, &alpha,
                  B, ldb, A, lda, &beta, C, N);
}

#if 0
template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::float16 alpha, const platform::float16 *A,
    const platform::float16 *B, platform::float16 beta,
    platform::float16 *C) const {
  // Note that hipblas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  // CUDA 7.5 does not support hipblasGemmEx, hence we fall back to use hgemm
  CUBlas<platform::float16>::GEMM(context_.hipblas_handle(), cuTransB, cuTransA,
                                  N, M, K, &h_alpha, B, ldb, A, lda,
                                  &h_beta, C, N);
}
#endif

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(bool transA, bool transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             int lda, const T *B, int ldb,
                                             T beta, T *C, int ldc) const {
  // Note that hipblas follows fortran order, so the order is different from
  // the cblas convention.
  hipblasOperation_t cuTransA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t cuTransB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  CUBlas<T>::GEMM(context_.hipblas_handle(), cuTransB, cuTransA, N, M, K, &alpha,
                  B, ldb, A, lda, &beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::AXPY(int n, T alpha, const T *x,
                                             T *y) const {
  CUBlas<T>::AXPY(context_.hipblas_handle(), n, &alpha, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMV(bool trans_a, int M, int N,
                                             T alpha, const T *A, const T *B,
                                             T beta, T *C) const {
  hipblasOperation_t cuTransA = !trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  CUBlas<T>::GEMV(context_.hipblas_handle(), cuTransA, N, M, &alpha, A, N, B, 1,
                  &beta, C, 1);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T *A, const T *B, T beta, T *C, int batchCount,
    int64_t strideA, int64_t strideB) const {
  // Note that hipblas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  hipblasOperation_t cuTransA =
      (transA == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  hipblasOperation_t cuTransB =
      (transB == CblasNoTrans) ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  const int64_t strideC = M * N;

  CUBlas<T>::GEMM_BATCH(context_.hipblas_handle(), cuTransB, cuTransA, N, M, K,
                        &alpha, B, ldb, strideB, A, lda, strideA, &beta, C, ldc,
                        strideC, batchCount);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
