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
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE(platform::dynload::cublasDgemm(args...));
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
};

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(const CBLAS_TRANSPOSE transA,
                                             const CBLAS_TRANSPOSE transB,
                                             const int M, const int N,
                                             const int K, const T alpha,
                                             const T *A, const T *B,
                                             const T beta, T *C) const {
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
    const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, const int M,
    const int N, const int K, const platform::float16 alpha,
    const platform::float16 *A, const platform::float16 *B,
    const platform::float16 beta, platform::float16 *C) const {
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
void Blas<platform::CUDADeviceContext>::GEMM(
    const bool transA, const bool transB, const int M, const int N, const int K,
    const T alpha, const T *A, const int lda, const T *B, const int ldb,
    const T beta, T *C, const int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = transB == false ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBlas<T>::GEMM(context_.cublas_handle(), cuTransB, cuTransA, N, M, K, &alpha,
                  B, ldb, A, lda, &beta, C, ldc);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
