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

#if defined(__MUSACC__)
#include <thrust/device_vector.h>
#endif
#include "glog/logging.h"
#include "paddle/utils/flags.h"

#include "paddle/phi/backends/dynload/mublas.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/funcs/math_function.h"

PHI_DECLARE_bool(enable_cublas_tensor_op_math);
PHI_DECLARE_bool(gemm_use_half_precision_compute_type);

namespace phi {
namespace funcs {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasScopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemmBatched(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mublasSgemmStridedBatched(args...));
  }

  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const float *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc) {
// Because the gcc 4.8 doesn't expand template parameter pack that
// appears in a lambda-expression, I can not use template parameter pack
// here.
    // VLOG(5) << "use_tensor_op_math: "
    //         << (dev_ctx->tensor_core_available() ? "True" : "False");
    // dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
    //   PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemmEx(handle,
    //                                                          transa,
    //                                                          transb,
    //                                                          m,
    //                                                          n,
    //                                                          k,
    //                                                          alpha,
    //                                                          A,
    //                                                          Atype,
    //                                                          lda,
    //                                                          B,
    //                                                          Btype,
    //                                                          ldb,
    //                                                          beta,
    //                                                          C,
    //                                                          Ctype,
    //                                                          ldc));
    // });
      PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasSgemmEx."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasStrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgetrfBatched(args...));
    PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasSgetrfBatched."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgetriBatched(args...));
    PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasSgetriBatched."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSmatinvBatched(args...));
    PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasSmatinvBatched."));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgetrsBatched(args...));
      PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasSgetrsBatched."));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasStrsmBatched(args...));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgemmBatched(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mublasDgemmStridedBatched(args...));
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args UNUSED) {
    PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasDgemmEx."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgetrfBatched(args...));
     PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasDgetrfBatched."));   
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgetriBatched(args...));
     PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasDgetriBatched."));       
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDmatinvBatched(args...));
     PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasDmatinvBatched."));       
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgetrsBatched(args...));
     PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasDgetrsBatched."));        
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDtrsmBatched(args...));
  }
};

template <>
struct CUBlas<phi::dtype::float16> {
  using float16 = phi::dtype::float16;

  static void GEMM(mublasHandle_t handle,
                   mublasOperation_t transa,
                   mublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float16 *alpha,
                   const float16 *A,
                   int lda,
                   const float16 *B,
                   int ldb,
                   const float16 *beta,
                   float16 *C,
                   int ldc) {
    // PADDLE_ENFORCE_GPU_SUCCESS(
    //     phi::dynload::mublasHgemm(handle,
    //                               transa,
    //                               transb,
    //                               m,
    //                               n,
    //                               k,
    //                               reinterpret_cast<const __half *>(alpha),
    //                               reinterpret_cast<const __half *>(A),
    //                               lda,
    //                               reinterpret_cast<const __half *>(B),
    //                               ldb,
    //                               reinterpret_cast<const __half *>(beta),
    //                               reinterpret_cast<__half *>(C),
    //                               ldc));
     PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasHgemm."));        
  }

  static void GEMM_BATCH(phi::GPUContext *dev_ctx,
                         mublasOperation_t transa,
                         mublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float *alpha,
                         const float16 **A,
                         musaDataType_t Atype,
                         int lda,
                         const float16 **B,
                         musaDataType_t Btype,
                         int ldb,
                         const float *beta,
                         float16 **C,
                         musaDataType_t Ctype,
                         int ldc,
                         int batchCount,
                         musaDataType_t computeType) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "mublasGemmBatchedEx is not supported"));
  }

  static void GEMM_STRIDED_BATCH(mublasHandle_t handle,
                                 mublasOperation_t transa,
                                 mublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float16 *alpha,
                                 const float16 *A,
                                 int lda,
                                 long long int strideA,  // NOLINT
                                 const float16 *B,       // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const float16 *beta,
                                 float16 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "mublasHgemmStridedBatched is not supported"));                                  
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasHgemmStridedBatched(
    //     handle,
    //     transa,
    //     transb,
    //     m,
    //     n,
    //     k,
    //     reinterpret_cast<const __half *>(alpha),
    //     reinterpret_cast<const __half *>(A),
    //     lda,
    //     strideA,
    //     reinterpret_cast<const __half *>(B),
    //     ldb,
    //     strideB,
    //     reinterpret_cast<const __half *>(beta),
    //     reinterpret_cast<__half *>(C),
    //     ldc,
    //     strideC,
    //     batchCount));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc,
                      musaDataType_t computeType) {
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");

    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            ldc,
                                                            computeType,
                                                            algo));
    });
  }
};

template <>
struct CUBlas<phi::dtype::complex<float>> {
  static void GEMV(mublasHandle_t handle,
                   mublasOperation_t transa,
                   int m,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   const phi::dtype::complex<float> *B,
                   int ldb,
                   const phi::dtype::complex<float> *beta,
                   phi::dtype::complex<float> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        reinterpret_cast<const muFloatComplex *>(B),
        ldb,
        reinterpret_cast<const muFloatComplex *>(beta),
        reinterpret_cast<muFloatComplex *>(C),
        ldc));
  }

  static void AXPY(mublasHandle_t handle,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *X,
                   const int incX,
                   phi::dtype::complex<float> *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCaxpy(
        handle,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(X),
        incX,
        reinterpret_cast<muFloatComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(mublasHandle_t handle,
                                 mublasOperation_t transa,
                                 mublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const phi::dtype::complex<float> *alpha,
                                 const phi::dtype::complex<float> *A,
                                 int lda,
                                 long long int strideA,                // NOLINT
                                 const phi::dtype::complex<float> *B,  // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const phi::dtype::complex<float> *beta,
                                 phi::dtype::complex<float> *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const muFloatComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const muFloatComplex *>(beta),
        reinterpret_cast<muFloatComplex *>(C),
        ldc,
        strideC,
        batchCount));
  }

  static void GEMM(mublasHandle_t handle,
                   mublasOperation_t transa,
                   mublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   const phi::dtype::complex<float> *B,
                   int ldb,
                   const phi::dtype::complex<float> *beta,
                   phi::dtype::complex<float> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        reinterpret_cast<const muFloatComplex *>(B),
        ldb,
        reinterpret_cast<const muFloatComplex *>(beta),
        reinterpret_cast<muFloatComplex *>(C),
        ldc));
  }

  static void TRSM(mublasHandle_t handle,
                   mublasSideMode_t side,
                   mublasFillMode_t uplo,
                   mublasOperation_t transa,
                   mublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::dtype::complex<float> *alpha,
                   const phi::dtype::complex<float> *A,
                   int lda,
                   phi::dtype::complex<float> *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        reinterpret_cast<muFloatComplex *>(B),
        ldb));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/muda/mublas/index.html#mublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc,
                      musaDataType_t computeType) {
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");

    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            ldc,
                                                            computeType,
                                                            algo));
    });
  }

  static void TRSM_BATCH(mublasHandle_t handle,
                         mublasSideMode_t side,
                         mublasFillMode_t uplo,
                         mublasOperation_t transa,
                         mublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::dtype::complex<float> *alpha,
                         const phi::dtype::complex<float> **A,
                         int lda,
                         phi::dtype::complex<float> **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex **>(A),
        lda,
        reinterpret_cast<muFloatComplex **>(B),
        ldb,
        batch_size));
  }
};

template <>
struct CUBlas<phi::dtype::complex<double>> {
  static void GEMV(mublasHandle_t handle,
                   mublasOperation_t transa,
                   int m,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   const phi::dtype::complex<double> *B,
                   int ldb,
                   const phi::dtype::complex<double> *beta,
                   phi::dtype::complex<double> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        reinterpret_cast<const muDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const muDoubleComplex *>(beta),
        reinterpret_cast<muDoubleComplex *>(C),
        ldc));
  }

  static void AXPY(mublasHandle_t handle,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *X,
                   const int incX,
                   phi::dtype::complex<double> *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZaxpy(
        handle,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(X),
        incX,
        reinterpret_cast<muDoubleComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(
      mublasHandle_t handle,
      mublasOperation_t transa,
      mublasOperation_t transb,
      int m,
      int n,
      int k,
      const phi::dtype::complex<double> *alpha,
      const phi::dtype::complex<double> *A,
      int lda,
      long long int strideA,                 // NOLINT
      const phi::dtype::complex<double> *B,  // NOLINT
      int ldb,
      long long int strideB,  // NOLINT
      const phi::dtype::complex<double> *beta,
      phi::dtype::complex<double> *C,
      int ldc,
      long long int strideC,  // NOLINT
      int batchCount) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const muDoubleComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const muDoubleComplex *>(beta),
        reinterpret_cast<muDoubleComplex *>(C),
        ldc,
        strideC,
        batchCount));
  }

  static void GEMM(mublasHandle_t handle,
                   mublasOperation_t transa,
                   mublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   const phi::dtype::complex<double> *B,
                   int ldb,
                   const phi::dtype::complex<double> *beta,
                   phi::dtype::complex<double> *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        reinterpret_cast<const muDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const muDoubleComplex *>(beta),
        reinterpret_cast<muDoubleComplex *>(C),
        ldc));
  }

  static void TRSM(mublasHandle_t handle,
                   mublasSideMode_t side,
                   mublasFillMode_t uplo,
                   mublasOperation_t transa,
                   mublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::dtype::complex<double> *alpha,
                   const phi::dtype::complex<double> *A,
                   int lda,
                   phi::dtype::complex<double> *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        reinterpret_cast<muDoubleComplex *>(B),
        ldb));
  }

  static void TRSM_BATCH(mublasHandle_t handle,
                         mublasSideMode_t side,
                         mublasFillMode_t uplo,
                         mublasOperation_t transa,
                         mublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::dtype::complex<double> *alpha,
                         const phi::dtype::complex<double> **A,
                         int lda,
                         phi::dtype::complex<double> **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex **>(A),
        lda,
        reinterpret_cast<muDoubleComplex **>(B),
        ldb,
        batch_size));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc,
                      musaDataType_t computeType) {
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");

    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            ldc,
                                                            computeType,
                                                            algo));
    });

  }
};

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  context_.CublasCall([&](mublasHandle_t handle) {
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
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {
  // // Note that cublas follows fortran order, so the order is different from
  // // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::float16>::GEMM_EX(&cuda_ctx,
                                       cuTransB,
                                       cuTransA,
                                       N,
                                       M,
                                       K,
                                       &h_alpha,
                                       B,
                                       MUSA_R_16F,
                                       ldb,
                                       A,
                                       MUSA_R_16F,
                                       lda,
                                       &h_beta,
                                       C,
                                       MUSA_R_16F,
                                       N,
                                       (musaDataType_t)0);//MUSA_R_32F https://jira.mthreads.com/browse/SW-37038
}




template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {
                                            PADDLE_THROW(phi::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::complex<float> alpha,
                                        const phi::dtype::complex<float> *A,
                                        const phi::dtype::complex<float> *B,
                                        phi::dtype::complex<float> beta,
                                        phi::dtype::complex<float> *C) const {  
                                          PADDLE_THROW(phi::errors::Unimplemented(
      "Blas::GEMM for dtype complex<double> is not supported on MUSA now!"));
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
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
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  thrust::complex<double> c_alpha =
      thrust::complex<double>(alpha.real, alpha.imag);
  thrust::complex<double> c_beta =
      thrust::complex<double>(beta.real, beta.imag);
  auto &cuda_ctx = const_cast<phi::GPUContext &>(context_);
  CUBlas<phi::dtype::complex<double>>::GEMM_EX(&cuda_ctx,
                                               cuTransB,
                                               cuTransA,
                                               N,
                                               M,
                                               K,
                                               &c_alpha,
                                               B,
                                               // Originally, this was MUSA_C_64F, but due to some bugs, it was necessary to manually specify a value
                                               // jira:https://jira.mthreads.com/browse/SW-37038
                                               (musaDataType_t)5,//MUSA_C_64F
                                               ldb,
                                               A,
                                               (musaDataType_t)5,//MUSA_C_64F
                                               lda,
                                               &c_beta,
                                               C,
                                               (musaDataType_t)5,//MUSA_C_64F
                                               N,
                                               (musaDataType_t)5);//MUSA_C_64F
}


template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(bool transA,
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
  mublasOperation_t cuTransA = transA ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t cuTransB = transB ? MUBLAS_OP_T : MUBLAS_OP_N;

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
                       (musaDataType_t)0,//MUSA_R_32F,
                       ldb,
                       A,
                       (musaDataType_t)0,//MUSA_R_32F,
                       lda,
                       &beta,
                       C,
                       (musaDataType_t)0,//MUSA_R_32F,
                       ldc);
  } else {
    context_.CublasCall([&](mublasHandle_t handle) {
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
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
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
  mublasOperation_t cuTransA = transA ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t cuTransB = transB ? MUBLAS_OP_T : MUBLAS_OP_N;

  context_.CublasCall([&](mublasHandle_t handle) {
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
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        int lda,
                                        const phi::dtype::bfloat16 *B,
                                        int ldb,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C,
                                        int ldc) const {
  PADDLE_THROW(phi::errors::Unimplemented(
      "Blas::GEMM for dtype bfloat16 is not supported on MUSA now!"));
}

template <>
template <typename T>
void Blas<phi::GPUContext>::AXPY(int n, T alpha, const T *x, T *y) const {
  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::SCAL(int n, const T alpha, T *x) const {
  context_.CublasCall(
      [&](mublasHandle_t handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::VCOPY(int n, const T *x, T *y) const {
  context_.CublasCall(
      [&](mublasHandle_t handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  mublasOperation_t cuTransA = !trans_a ? MUBLAS_OP_T : MUBLAS_OP_N;

  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GEMV(handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::dtype::float16 alpha,
                                        const phi::dtype::float16 *A,
                                        const phi::dtype::float16 *B,
                                        phi::dtype::float16 beta,
                                        phi::dtype::float16 *C) const {
  // Because cublas doesn't support half gemv, we use cublasHgemm to achieve it.
  if (trans_a) {
    this->template GEMM<phi::dtype::float16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::dtype::float16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::dtype::bfloat16 alpha,
                                        const phi::dtype::bfloat16 *A,
                                        const phi::dtype::bfloat16 *B,
                                        phi::dtype::bfloat16 beta,
                                        phi::dtype::bfloat16 *C) const {
  // Because cublas doesn't support bfloat gemv, we use cublasHgemm to achieve
  // it.
  if (trans_a) {
    this->template GEMM<phi::dtype::bfloat16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::dtype::bfloat16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}


template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  const int64_t strideC = M * N;
    context_.CublasCall([&](mublasHandle_t handle) {
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
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  // int lda = (transA == CblasNoTrans) ? K : M;
  // int ldb = (transB == CblasNoTrans) ? N : K;
  // int ldc = N;
  // mublasOperation_t cuTransA =
  //     (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  // mublasOperation_t cuTransB =
  //     (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  // const int64_t strideC = M * N;

  // float h_alpha = static_cast<float>(alpha);
  // float h_beta = static_cast<float>(beta);

  // mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  // bool use_tensor_op_math = context_.tensor_core_available();
  // if (use_tensor_op_math) {
  //   algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
  // }
  // VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  // context_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
  //   PADDLE_ENFORCE_GPU_SUCCESS(
  //       phi::dynload::mublasGemmStridedBatchedEx(handle,
  //                                                cuTransB,
  //                                                cuTransA,
  //                                                N,
  //                                                M,
  //                                                K,
  //                                                &h_alpha,
  //                                                B,
  //                                                MUSA_R_16BF,
  //                                                ldb,
  //                                                strideB,
  //                                                A,
  //                                                MUSA_R_16BF,
  //                                                lda,
  //                                                strideA,
  //                                                &h_beta,
  //                                                C,
  //                                                MUSA_R_16BF,
  //                                                ldc,
  //                                                strideC,
  //                                                batchCount,
  //                                                MUBLAS_COMPUTE_32F,
  //                                                algo));
  // });
       PADDLE_THROW(
        phi::errors::Unimplemented("murrently there are not mublasGemmStridedBatchedEx."));   
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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

#if defined(__MUSACC__)
template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               double alpha,
                                               const double **A,
                                               const double **B,
                                               double beta,
                                               double **C,
                                               int batchCount) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  thrust::device_vector<const double *> A_ptr(A, A + batchCount);
  thrust::device_vector<const double *> B_ptr(B, B + batchCount);
  thrust::device_vector<double *> C_ptr(C, C + batchCount);

  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<double>::GEMM_BATCH(handle,
                               cuTransB,
                               cuTransA,
                               N,
                               M,
                               K,
                               &alpha,
                               B_ptr.data().get(),
                               ldb,
                               A_ptr.data().get(),
                               lda,
                               &beta,
                               C_ptr.data().get(),
                               ldc,
                               batchCount);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               const float **A,
                                               const float **B,
                                               float beta,
                                               float **C,
                                               int batchCount) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  thrust::device_vector<const float *> A_ptr(A, A + batchCount);
  thrust::device_vector<const float *> B_ptr(B, B + batchCount);
  thrust::device_vector<float *> C_ptr(C, C + batchCount);

  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<float>::GEMM_BATCH(handle,
                              cuTransB,
                              cuTransA,
                              N,
                              M,
                              K,
                              &alpha,
                              B_ptr.data().get(),
                              ldb,
                              A_ptr.data().get(),
                              lda,
                              &beta,
                              C_ptr.data().get(),
                              ldc,
                              batchCount);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  PADDLE_THROW(phi::errors::Unimplemented(
      "Blas::BatchedGEMM for dtype float16 is not supported on MUSA now!"));
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  PADDLE_THROW(phi::errors::Unimplemented(
      "Blas::BatchedGEMM for bfloat16 is not supported on MUSA now!"));
}
#endif
template <>
template <typename T>
void Blas<phi::GPUContext>::TRSM(CBLAS_SIDE side,
                                 CBLAS_UPLO uplo,
                                 CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 T *B,
                                 int ldb) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  mublasSideMode_t cuSide =
      (side == CblasLeft) ? MUBLAS_SIDE_RIGHT : MUBLAS_SIDE_LEFT;
  mublasFillMode_t cuUplo =
      (uplo == CblasLower) ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasDiagType_t cuDiag =
      (diag == CblasUnit) ? MUBLAS_DIAG_UNIT : MUBLAS_DIAG_NON_UNIT;

  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::TRSM(
        handle, cuSide, cuUplo, cuTransA, cuDiag, N, M, &alpha, A, lda, B, ldb);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRF(
    int n, T **a, int *ipiv, int *info, int batch_size) const {
  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GETRF_BATCH(handle, n, a, n, ipiv, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRI(int n,
                                         const T **a,
                                         const int *ipiv,
                                         T **a_inv,
                                         int *info,
                                         int batch_size) const {
  PADDLE_ENFORCE_NE(
      a_inv,
      a,
      phi::errors::InvalidArgument(
          "cuBLAS fuction 'cublas<S/D>getrfBatched' cannot be executed "
          "in-place. The memory space of output matrix (address: %p) cannot "
          "overlap memory space of input matrix (address: %p).",
          a_inv,
          a));
  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GETRI_BATCH(handle, n, a, n, ipiv, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedMatInv(
    int n, const T **a, T **a_inv, int *info, int batch_size) const {
  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::MATINV_BATCH(handle, n, a, n, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRS(CBLAS_TRANSPOSE trans,
                                         int n,
                                         int nrhs,
                                         const T **a,
                                         int lda,
                                         int *ipiv,
                                         T **b,
                                         int ldb,
                                         int *info,
                                         int batch_size) const {
  // use CUBLAS_OP_C (conjugate transpose) for complex
  mublasOperation_t cuTrans =
      (trans == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GETRS_BATCH(
        handle, cuTrans, n, nrhs, a, lda, ipiv, b, ldb, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedTRSM(CBLAS_SIDE side,
                                        CBLAS_UPLO uplo,
                                        CBLAS_TRANSPOSE transA,
                                        CBLAS_DIAG diag,
                                        int M,
                                        int N,
                                        T alpha,
                                        const T **A,
                                        int lda,
                                        T **B,
                                        int ldb,
                                        int batch_size) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  mublasSideMode_t cuSide =
      (side == CblasLeft) ? MUBLAS_SIDE_RIGHT : MUBLAS_SIDE_LEFT;
  mublasFillMode_t cuUplo =
      (uplo == CblasLower) ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasDiagType_t cuDiag =
      (diag == CblasUnit) ? MUBLAS_DIAG_UNIT : MUBLAS_DIAG_NON_UNIT;

  context_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::TRSM_BATCH(handle,
                          cuSide,
                          cuUplo,
                          cuTransA,
                          cuDiag,
                          N,
                          M,
                          &alpha,
                          A,
                          lda,
                          B,
                          ldb,
                          batch_size);
  });
}

}  // namespace funcs
}  // namespace phi
