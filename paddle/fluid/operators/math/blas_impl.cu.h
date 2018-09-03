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
#include "paddle/fluid/platform/dynload/cublas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasSaxpy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE(platform::dynload::cublasSgemmStridedBatched(args...));
#else
    PADDLE_THROW("SgemmStridedBatched is not supported on cuda <= 7.5");
#endif
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasDaxpy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE(platform::dynload::cublasDgemmStridedBatched(args...));
#else
    PADDLE_THROW("DgemmStridedBatched is not supported on cuda <= 7.5");
#endif
  }
};

template <>
struct CUBlas<platform::float16> {
  using float16 = platform::float16;

  static void GEMM(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float16 *alpha, const float16 *A, int lda,
                   const float16 *B, int ldb, const float16 *beta, float16 *C,
                   int ldc) {
    PADDLE_ENFORCE(
        platform::dynload::cublasHgemm(handle, transa, transb, m, n, k,
                                       reinterpret_cast<const __half *>(alpha),
                                       reinterpret_cast<const __half *>(A), lda,
                                       reinterpret_cast<const __half *>(B), ldb,
                                       reinterpret_cast<const __half *>(beta),
                                       reinterpret_cast<__half *>(C), ldc));
  }

  static void GEMM_BATCH(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, int m, int n, int k,
                         const float16 *alpha, const float16 *A, int lda,
                         long long int strideA, const float16 *B,  // NOLINT
                         int ldb, long long int strideB,           // NOLINT
                         const float16 *beta, float16 *C, int ldc,
                         long long int strideC,  // NOLINT
                         int batchCount) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE(platform::dynload::cublasHgemmStridedBatched(
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

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                             CBLAS_TRANSPOSE transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             const T *B, T beta, T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  CUBlas<T>::GEMM(context_.cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha,
                  B, ldb, A, lda, &beta, C, N);
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::float16 alpha, const platform::float16 *A,
    const platform::float16 *B, platform::float16 beta,
    platform::float16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(context_.GetComputeCapability(), 53,
                    "cublas fp16 gemm requires GPU compute capability >= 53");

#if CUDA_VERSION >= 8000
  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
  if (context_.GetComputeCapability() >= 70) {
    PADDLE_ENFORCE(platform::dynload::cublasSetMathMode(
        context_.cublas_handle(), CUBLAS_TENSOR_OP_MATH));
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  } else {
    PADDLE_ENFORCE(platform::dynload::cublasSetMathMode(
        context_.cublas_handle(), CUBLAS_DEFAULT_MATH));
  }
#endif  // CUDA_VERSION >= 9000

  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  PADDLE_ENFORCE(platform::dynload::cublasGemmEx(
      context_.cublas_handle(), cuTransB, cuTransA, N, M, K, &h_alpha, B,
      CUDA_R_16F, ldb, A, CUDA_R_16F, lda, &h_beta, C, CUDA_R_16F, N,
      CUDA_R_32F, algo));
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm
  CUBlas<platform::float16>::GEMM(context_.cublas_handle(), cuTransB, cuTransA,
                                  N, M, K, &h_alpha, h_B, ldb, h_A, lda,
                                  &h_beta, h_C, N);
#endif  // CUDA_VERSION >= 8000
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(bool transA, bool transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             int lda, const T *B, int ldb,
                                             T beta, T *C, int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBlas<T>::GEMM(context_.cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha,
                  B, ldb, A, lda, &beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::AXPY(int n, T alpha, const T *x,
                                             T *y) const {
  CUBlas<T>::AXPY(context_.cublas_handle(), n, &alpha, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMV(bool trans_a, int M, int N,
                                             T alpha, const T *A, const T *B,
                                             T beta, T *C) const {
  cublasOperation_t cuTransA = !trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

  CUBlas<T>::GEMV(context_.cublas_handle(), cuTransA, N, M, &alpha, A, N, B, 1,
                  &beta, C, 1);
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T *A, const T *B, T beta, T *C, int batchCount,
    int64_t strideA, int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  CUBlas<T>::GEMM_BATCH(context_.cublas_handle(), cuTransB, cuTransA, N, M, K,
                        &alpha, B, ldb, strideB, A, lda, strideA, &beta, C, ldc,
                        strideC, batchCount);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
