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

#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/backends/dynload/cublasLt.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"

DECLARE_bool(enable_cublas_tensor_op_math);
DECLARE_bool(gemm_use_half_precision_compute_type);

namespace phi {
namespace funcs {

template <typename T>
struct CUBlasLt;

template <>
struct CUBlasLt<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmul(args...));
  }
};

template <>
struct CUBlasLt<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasLtMatmul(args...));
  }
};













/*







template <>
template <typename T>
void BlasLt<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
    CUBlas<T>::GEMM_EX(&cuda_ctx,
                       cuTransB,
                       cuTransA,
                       N,
                       M,
                       K,
                       &alpha,
                       B,
                       CUDA_R_32F,
                       ldb,
                       A,
                       CUDA_R_32F,
                       lda,
                       &beta,
                       C,
                       CUDA_R_32F,
                       N);
  } else {
#endif  // CUDA_VERSION >= 8000
    context_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM(handle,
                      cuTransB,
                      cuTransA,
                      N,
                      M,
                      K,
                      &alpha,
                      B,
                      ldb,
                      A,
                      lda,
                      &beta,
                      C,
                      N);
    });

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      phi::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::float16>::GEMM_EX(&cuda_ctx,
                                       cuTransB,
                                       cuTransA,
                                       N,
                                       M,
                                       K,
                                       &h_alpha,
                                       B,
                                       CUDA_R_16F,
                                       ldb,
                                       A,
                                       CUDA_R_16F,
                                       lda,
                                       &h_beta,
                                       C,
                                       CUDA_R_16F,
                                       N,
                                       CUDA_R_32F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::float16>::GEMM(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &h_alpha,
                                      h_B,
                                      ldb,
                                      h_A,
                                      lda,
                                      &h_beta,
                                      h_C,
                                      N);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      80,
      phi::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = context_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
                                                          cuTransB,
                                                          cuTransA,
                                                          N,
                                                          M,
                                                          K,
                                                          &h_alpha,
                                                          B,
                                                          CUDA_R_16BF,
                                                          ldb,
                                                          A,
                                                          CUDA_R_16BF,
                                                          lda,
                                                          &h_beta,
                                                          C,
                                                          CUDA_R_16BF,
                                                          N,
                                                          CUDA_R_32F,
                                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(phi::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<float> alpha,
                                        const phi::dtype::complex<float> *A,
                                        const phi::dtype::complex<float> *B,
                                        phi::dtype::complex<float> beta,
                                        phi::dtype::complex<float> *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      phi::errors::InvalidArgument(
          "cublas complex64 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<float> c_alpha =
      thrust::complex<float>(alpha.real, alpha.imag);
  thrust::complex<float> c_beta = thrust::complex<float>(beta.real, beta.imag);

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<float>>::GEMM_EX(&cuda_ctx,
                                              cuTransB,
                                              cuTransA,
                                              N,
                                              M,
                                              K,
                                              &c_alpha,
                                              B,
                                              CUDA_C_32F,
                                              ldb,
                                              A,
                                              CUDA_C_32F,
                                              lda,
                                              &c_beta,
                                              C,
                                              CUDA_C_32F,
                                              N,
                                              CUDA_C_32F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::complex<float>>::GEMM(handle,
                                             cuTransB,
                                             cuTransA,
                                             N,
                                             M,
                                             K,
                                             &c_alpha,
                                             h_B,
                                             ldb,
                                             h_A,
                                             lda,
                                             &c_beta,
                                             h_C,
                                             N);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<double> alpha,
                                        const phi::dtype::complex<double> *A,
                                        const phi::dtype::complex<double> *B,
                                        phi::dtype::complex<double> beta,
                                        phi::dtype::complex<double> *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(),
      53,
      phi::errors::InvalidArgument(
          "cublas complex128 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<double> c_alpha =
      thrust::complex<double>(alpha.real, alpha.imag);
  thrust::complex<double> c_beta =
      thrust::complex<double>(beta.real, beta.imag);

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<double>>::GEMM_EX(&cuda_ctx,
                                               cuTransB,
                                               cuTransA,
                                               N,
                                               M,
                                               K,
                                               &c_alpha,
                                               B,
                                               CUDA_C_64F,
                                               ldb,
                                               A,
                                               CUDA_C_64F,
                                               lda,
                                               &c_beta,
                                               C,
                                               CUDA_C_64F,
                                               N,
                                               CUDA_C_64F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::complex<double>>::GEMM(handle,
                                              cuTransB,
                                              cuTransA,
                                              N,
                                              M,
                                              K,
                                              &c_alpha,
                                              h_B,
                                              ldb,
                                              h_A,
                                              lda,
                                              &c_beta,
                                              h_C,
                                              N);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <typename T>
void BlasLt<phi::GPUContext>::GEMM(bool transA,
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
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
    CUBlas<T>::GEMM_EX(&cuda_ctx,
                       cuTransB,
                       cuTransA,
                       N,
                       M,
                       K,
                       &alpha,
                       B,
                       CUDA_R_32F,
                       ldb,
                       A,
                       CUDA_R_32F,
                       lda,
                       &beta,
                       C,
                       CUDA_R_32F,
                       ldc);
  } else {
#endif  // CUDA_VERSION >= 8000

    context_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM(handle,
                      cuTransB,
                      cuTransA,
                      N,
                      M,
                      K,
                      &alpha,
                      B,
                      ldb,
                      A,
                      lda,
                      &beta,
                      C,
                      ldc);
    });

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        int lda,
                                        const phi::dtype::float16 *B,
                                        int ldb,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C,
                                        int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::dtype::float16>::GEMM(handle,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &alpha,
                                      B,
                                      ldb,
                                      A,
                                      lda,
                                      &beta,
                                      C,
                                      ldc);
  });
}

template <>
template <typename T>
void BlasLt<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T *A,
                                        const T *B,
                                        T beta,
                                        T *C,
                                        int batchCount,
                                        int64_t strideA,
                                        int64_t strideB) const {
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

#if CUDA_VERSION >= 9010
  if ((FLAGS_enable_cublas_tensor_op_math && (std::is_same<T, float>::value)) ||
      std::is_same<T, phi::dtype::float16>::value) {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
    bool use_tensor_op_math = context_.tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
    VLOG(4) << "use_half_precision_compute_type: "
            << FLAGS_gemm_use_half_precision_compute_type;

    auto fp = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_16F;
#if CUDA_VERSION >= 11000
    auto compute_type = CUBLAS_COMPUTE_32F;
#else
    auto compute_type = CUDA_R_32F;
#endif

    float h_alpha = static_cast<float>(alpha);
    float h_beta = static_cast<float>(beta);
    void *a = static_cast<void *>(&h_alpha);
    void *b = static_cast<void *>(&h_beta);
    // set ComputeType as CUDA_R_32F for fp16, for better accuracy
    if (FLAGS_gemm_use_half_precision_compute_type == true &&
        std::is_same<T, phi::dtype::float16>::value) {
      a = static_cast<void *>(&alpha);
      b = static_cast<void *>(&beta);
#if CUDA_VERSION >= 11000
      compute_type = CUBLAS_COMPUTE_16F;
#else
      compute_type = CUDA_R_16F;
#endif
    }

    context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   N,
                                                   M,
                                                   K,
                                                   a,
                                                   B,
                                                   fp,
                                                   ldb,
                                                   strideB,
                                                   A,
                                                   fp,
                                                   lda,
                                                   strideA,
                                                   b,
                                                   C,
                                                   fp,
                                                   ldc,
                                                   strideC,
                                                   batchCount,
                                                   compute_type,
                                                   algo));
    });
  } else {
#endif  // CUDA_VERSION >= 9010

    context_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                    cuTransB,
                                    cuTransA,
                                    N,
                                    M,
                                    K,
                                    &alpha,
                                    B,
                                    ldb,
                                    strideB,
                                    A,
                                    lda,
                                    strideA,
                                    &beta,
                                    C,
                                    ldc,
                                    strideC,
                                    batchCount);
    });

#if CUDA_VERSION >= 9010
  }
#endif  // CUDA_VERSION >= 9010
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::bfloat16 alpha,
                                               const phi::dtype::bfloat16 *A,
                                               const phi::dtype::bfloat16 *B,
                                               phi::dtype::bfloat16 beta,
                                               phi::dtype::bfloat16 *C,
                                               int batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
#if CUDA_VERSION >= 11000
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

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
  bool use_tensor_op_math = context_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                 cuTransB,
                                                 cuTransA,
                                                 N,
                                                 M,
                                                 K,
                                                 &h_alpha,
                                                 B,
                                                 CUDA_R_16BF,
                                                 ldb,
                                                 strideB,
                                                 A,
                                                 CUDA_R_16BF,
                                                 lda,
                                                 strideA,
                                                 &h_beta,
                                                 C,
                                                 CUDA_R_16BF,
                                                 ldc,
                                                 strideC,
                                                 batchCount,
                                                 CUBLAS_COMPUTE_32F,
                                                 algo));
  });
#else
  // raise error
  PADDLE_THROW(phi::errors::Unimplemented(
      "cublasGemmStridedBatchedEx with bfloat16 is not supported on cuda <= "
      "11"));
#endif  // CUDA_VERSION >= 11000
}

template <>
template <typename T>
void BlasLt<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::float16 alpha,
                                               const phi::dtype::float16 **A,
                                               const phi::dtype::float16 **B,
                                               phi::dtype::float16 beta,
                                               phi::dtype::float16 **C,
                                               int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<phi::dtype::float16>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

template <>
template <>
inline void BlasLt<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::dtype::bfloat16 alpha,
                                               const phi::dtype::bfloat16 **A,
                                               const phi::dtype::bfloat16 **B,
                                               phi::dtype::bfloat16 beta,
                                               phi::dtype::bfloat16 **C,
                                               int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<phi::dtype::bfloat16>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}


*/
}  // namespace funcs
}  // namespace phi
