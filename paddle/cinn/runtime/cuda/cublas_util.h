// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <cublas_v2.h>

#include "glog/logging.h"
#include "paddle/cinn/common/type.h"

namespace cinn {
namespace runtime {
namespace cuda {

inline cublasStatus_t cublasGemm(cudaDataType_t dtype,
                                 cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 float alpha,
                                 const void *A,
                                 int lda,
                                 const void *B,
                                 int ldb,
                                 float beta,
                                 void *C,
                                 int ldc) {
  if (dtype == CUDA_R_32F) {
    return cublasSgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       reinterpret_cast<const float *>(&alpha),
                       reinterpret_cast<const float *>(A),
                       lda,
                       reinterpret_cast<const float *>(B),
                       ldb,
                       reinterpret_cast<const float *>(&beta),
                       reinterpret_cast<float *>(C),
                       ldc);
  } else if (dtype == CUDA_R_64F) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    return cublasDgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       &alpha_fp64,
                       reinterpret_cast<const double *>(A),
                       lda,
                       reinterpret_cast<const double *>(B),
                       ldb,
                       &beta_fp64,
                       reinterpret_cast<double *>(C),
                       ldc);
  } else if (dtype == CUDA_R_16F) {
#if CUDA_VERSION >= 11000
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        &alpha,
                        A,
                        CUDA_R_16F,
                        lda,
                        B,
                        CUDA_R_16F,
                        ldb,
                        &beta,
                        C,
                        CUDA_R_16F,
                        ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    cinn::common::float16 alpha_fp16{alpha};
    cinn::common::float16 beta_fp16{beta};
    return cublasHgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       reinterpret_cast<const __half *>(&alpha_fp16),
                       reinterpret_cast<const __half *>(A),
                       lda,
                       reinterpret_cast<const __half *>(B),
                       ldb,
                       reinterpret_cast<const __half *>(&beta_fp16),
                       reinterpret_cast<__half *>(C),
                       ldc);
#endif
  } else if (dtype == CUDA_R_16BF) {
#if CUDA_VERSION >= 11000
    return cublasGemmEx(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        &alpha,
                        A,
                        CUDA_R_16BF,
                        lda,
                        B,
                        CUDA_R_16BF,
                        ldb,
                        &beta,
                        C,
                        CUDA_R_16BF,
                        ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    PADDLE_THROW(::common::errors::Fatal(
        "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));
#endif
  }
  PADDLE_THROW(
      ::common::errors::InvalidArgument("Unsupported cublasGemm precision."));
}

inline cublasStatus_t cublasGemmStridedBatched(cudaDataType_t dtype,
                                               cublasHandle_t handle,
                                               cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               int m,
                                               int n,
                                               int k,
                                               float alpha,
                                               const void *A,
                                               int lda,
                                               int64_t strideA,
                                               const void *B,
                                               int ldb,
                                               int64_t strideB,
                                               float beta,
                                               void *C,
                                               int ldc,
                                               int64_t strideC,
                                               int batchCount) {
  if (dtype == CUDA_R_32F) {
    return cublasSgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     reinterpret_cast<const float *>(&alpha),
                                     reinterpret_cast<const float *>(A),
                                     lda,
                                     strideA,
                                     reinterpret_cast<const float *>(B),
                                     ldb,
                                     strideB,
                                     reinterpret_cast<const float *>(&beta),
                                     reinterpret_cast<float *>(C),
                                     ldc,
                                     strideC,
                                     batchCount);
  } else if (dtype == CUDA_R_64F) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    return cublasDgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     &alpha_fp64,
                                     reinterpret_cast<const double *>(A),
                                     lda,
                                     strideA,
                                     reinterpret_cast<const double *>(B),
                                     ldb,
                                     strideB,
                                     &beta_fp64,
                                     reinterpret_cast<double *>(C),
                                     ldc,
                                     strideC,
                                     batchCount);
  } else if (dtype == CUDA_R_16F) {
#if CUDA_VERSION >= 11000
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      A,
                                      CUDA_R_16F,
                                      lda,
                                      strideA,
                                      B,
                                      CUDA_R_16F,
                                      ldb,
                                      strideB,
                                      &beta,
                                      C,
                                      CUDA_R_16F,
                                      ldc,
                                      strideC,
                                      batchCount,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    cinn::common::float16 alpha_fp16{alpha};
    cinn::common::float16 beta_fp16{beta};
    return cublasHgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const __half *>(&alpha_fp16),
        reinterpret_cast<const __half *>(A),
        lda,
        strideA,
        reinterpret_cast<const __half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const __half *>(&beta_fp16),
        reinterpret_cast<__half *>(C),
        ldc,
        strideC,
        batchCount);
#endif
  } else if (dtype == CUDA_R_16BF) {
#if CUDA_VERSION >= 11000
    return cublasGemmStridedBatchedEx(handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      A,
                                      CUDA_R_16BF,
                                      lda,
                                      strideA,
                                      B,
                                      CUDA_R_16BF,
                                      ldb,
                                      strideB,
                                      &beta,
                                      C,
                                      CUDA_R_16BF,
                                      ldc,
                                      strideC,
                                      batchCount,
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    PADDLE_THROW(::common::errors::InvalidArgument(
        "cublasGemmStridedBatched with bfloat16 is not supported on "
        "cuda <= 11"));
#endif
  }
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Unsupported cublasGemmStridedBatched precision."));
}

inline cublasStatus_t cublasGemmBatched(cudaDataType_t dtype,
                                        cublasHandle_t handle,
                                        cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        int m,
                                        int n,
                                        int k,
                                        float alpha,
                                        void **A,
                                        int lda,
                                        void **B,
                                        int ldb,
                                        float beta,
                                        void **C,
                                        int ldc,
                                        int batchCount) {
  if (dtype == CUDA_R_32F) {
    return cublasSgemmBatched(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              &alpha,
                              reinterpret_cast<float **>(A),
                              lda,
                              reinterpret_cast<float **>(B),
                              ldb,
                              &beta,
                              reinterpret_cast<float **>(C),
                              ldc,
                              batchCount);
  } else if (dtype == CUDA_R_64F) {
    const double alpha_fp64 = static_cast<double>(alpha);
    const double beta_fp64 = static_cast<double>(beta);
    return cublasDgemmBatched(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              &alpha_fp64,
                              reinterpret_cast<double **>(A),
                              lda,
                              reinterpret_cast<double **>(B),
                              ldb,
                              &beta_fp64,
                              reinterpret_cast<double **>(C),
                              ldc,
                              batchCount);
  } else if (dtype == CUDA_R_16F) {
#if CUDA_VERSION >= 11000
    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               &alpha,
                               A,
                               CUDA_R_16F,
                               lda,
                               B,
                               CUDA_R_16F,
                               ldb,
                               &beta,
                               C,
                               CUDA_R_16F,
                               ldc,
                               batchCount,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    __half alpha_fp16{alpha};
    __half beta_fp16{beta};
    return cublasHgemmBatched(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              &alpha_fp16,
                              reinterpret_cast<__half **>(A),
                              lda,
                              reinterpret_cast<__half **>(B),
                              ldb,
                              &beta_fp16,
                              reinterpret_cast<__half **>(C),
                              ldc,
                              batchCount);
#endif
  } else if (dtype == CUDA_R_16BF) {
#if CUDA_VERSION >= 11000
    return cublasGemmBatchedEx(handle,
                               transa,
                               transb,
                               m,
                               n,
                               k,
                               &alpha,
                               A,
                               CUDA_R_16BF,
                               lda,
                               B,
                               CUDA_R_16BF,
                               ldb,
                               &beta,
                               C,
                               CUDA_R_16BF,
                               ldc,
                               batchCount,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    PADDLE_THROW(::common::errors::Fatal(
        "cublasGemmBatched with bfloat16 is not supported on cuda <= 11"));
#endif
  }
  PADDLE_THROW(::common::errors::InvalidArgument(
      "Unsupported cublasGemmBatched precision."));
}

}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
