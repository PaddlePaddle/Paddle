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

#if defined(__NVCC__)
#include <thrust/device_vector.h>
#endif
#include <utility>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/scope_guard.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define INT_MAX_VALUE 2147483647

COMMON_DECLARE_bool(cublas_allow_tf32);
COMMON_DECLARE_bool(gemm_use_half_precision_compute_type);
COMMON_DECLARE_bool(use_legacy_gemm);

namespace phi {
namespace funcs {

namespace detail {

template <typename Callback>
inline void CublasCallWithTF32MathMode(const phi::GPUContext *dev_ctx,
                                       Callback &&callback) {
  dev_ctx->CublasCall([&](cublasHandle_t handle) {
#if CUDA_VERSION >= 11000
    cublasMath_t previous_mode = CUBLAS_DEFAULT_MATH;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasGetMathMode(handle, &previous_mode));
    const cublasMath_t math_mode = FLAGS_cublas_allow_tf32
                                       ? CUBLAS_TF32_TENSOR_OP_MATH
                                       : CUBLAS_DEFAULT_MATH;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasSetMathMode(handle, math_mode));
    DEFINE_PADDLE_SCOPE_GUARD([&] {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasSetMathMode(handle, previous_mode));
    });
#else
    PADDLE_ENFORCE_EQ(!FLAGS_cublas_allow_tf32,
                      true,
                      common::errors::Unimplemented(
                          "TF32 math mode requires CUDA 11.0 or later."));
#endif
    callback(handle);
  });
}

template <typename T, typename Callback>
inline void CublasCallForGemmEx(const phi::GPUContext *dev_ctx,
                                Callback &&callback) {
  if constexpr (std::is_same<T, float>::value) {
    CublasCallWithTF32MathMode(dev_ctx, std::forward<Callback>(callback));
  } else {
    dev_ctx->TensorCoreCublasCallIfAvailable(std::forward<Callback>(callback));
  }
}

inline void GemmEx(phi::GPUContext *dev_ctx,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const void *alpha,
                   const void *A,
                   cudaDataType_t Atype,
                   int lda,
                   const void *B,
                   cudaDataType_t Btype,
                   int ldb,
                   const void *beta,
                   void *C,
                   cudaDataType_t Ctype,
                   int ldc,
                   cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
  bool use_tensor_op_math = dev_ctx->tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

  dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx(handle,
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
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
}

inline void GemmEx64(phi::GPUContext *dev_ctx,
                     cublasOperation_t transa,
                     cublasOperation_t transb,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     const void *alpha,
                     const void *A,
                     cudaDataType_t Atype,
                     int64_t lda,
                     const void *B,
                     cudaDataType_t Btype,
                     int64_t ldb,
                     const void *beta,
                     void *C,
                     cudaDataType_t Ctype,
                     int64_t ldc,
                     cublasComputeType_t computeType) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
  bool use_tensor_op_math = dev_ctx->tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
  dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmEx_64(handle,
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
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx_64 is not supported on cuda < 12.3"));
#endif
}

}  // namespace detail

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemm(args...));
  }

  template <typename... ARGS>
  static void GEMM_64(ARGS... args) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemm_v2_64(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSaxpy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMV_64(ARGS... args) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemv_v2_64(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMV requires CUDA 12.3 or later on Linux."));
#endif
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemmBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "SgemmBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasSgemmStridedBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "SgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrfBatched(args...));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetriBatched(args...));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSmatinvBatched(args...));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgetrsBatched(args...));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasStrsmBatched(args...));
  }

  template <typename... ARGS>
  static void DOT(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSdot_v2(args...));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemm(args...));
  }

  template <typename... ARGS>
  static void GEMM_64(ARGS... args) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemm_v2_64(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDaxpy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMV_64(ARGS... args) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemv_v2_64(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMV requires CUDA 12.3 or later on Linux."));
#endif
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemmBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "DgemmBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasDgemmStridedBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "DgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetrfBatched(args...));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetriBatched(args...));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDmatinvBatched(args...));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgetrsBatched(args...));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDtrsmBatched(args...));
  }

  template <typename... ARGS>
  static void DOT(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDdot_v2(args...));
  }
};

template <>
struct CUBlas<phi::float16> {
  static void GEMM(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::float16 *alpha,
                   const phi::float16 *A,
                   int lda,
                   const phi::float16 *B,
                   int ldb,
                   const phi::float16 *beta,
                   phi::float16 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasHgemm(handle,
                                  transa,
                                  transb,
                                  m,
                                  n,
                                  k,
                                  reinterpret_cast<const __half *>(alpha),
                                  reinterpret_cast<const __half *>(A),
                                  lda,
                                  reinterpret_cast<const __half *>(B),
                                  ldb,
                                  reinterpret_cast<const __half *>(beta),
                                  reinterpret_cast<__half *>(C),
                                  ldc));
  }

#if defined(__NVCC__)
  static void GEMM_BATCH(phi::GPUContext *dev_ctx,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float *alpha,
                         const float16 **A,
                         cudaDataType_t Atype,
                         int lda,
                         const float16 **B,
                         cudaDataType_t Btype,
                         int ldb,
                         const float *beta,
                         float16 **C,
                         cudaDataType_t Ctype,
                         int ldc,
                         int batchCount,
                         cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000
    thrust::device_vector<const void *> A_ptr(A, A + batchCount);
    thrust::device_vector<const void *> B_ptr(B, B + batchCount);
    thrust::device_vector<void *> C_ptr(C, C + batchCount);
    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmBatchedEx(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A_ptr.data().get(),
                                            Atype,
                                            lda,
                                            B_ptr.data().get(),
                                            Btype,
                                            ldb,
                                            beta,
                                            C_ptr.data().get(),
                                            Ctype,
                                            ldc,
                                            batchCount,
                                            computeType,
                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmBatchedEx is not supported on cuda <= 7.5"));
#endif
  }
#endif

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
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
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasHgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const __half *>(alpha),
        reinterpret_cast<const __half *>(A),
        lda,
        strideA,
        reinterpret_cast<const __half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const __half *>(beta),
        reinterpret_cast<__half *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "HgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      cudaDataType_t Atype,
                      int lda,
                      const void *B,
                      cudaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      cudaDataType_t Ctype,
                      int ldc,
                      cudaDataType_t computeType) {
    detail::GemmEx(dev_ctx,
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
                   computeType);
  }

  static void GEMM_EX_64(phi::GPUContext *dev_ctx,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         const void *alpha,
                         const void *A,
                         cudaDataType_t Atype,
                         int64_t lda,
                         const void *B,
                         cudaDataType_t Btype,
                         int64_t ldb,
                         const void *beta,
                         void *C,
                         cudaDataType_t Ctype,
                         int64_t ldc,
                         cublasComputeType_t computeType) {
    detail::GemmEx64(dev_ctx,
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
                     computeType);
  }

  static void DOT(cublasHandle_t handle,
                  int n,
                  const phi::float16 *x,
                  const int incx,
                  const phi::float16 *y,
                  const int incy,
                  phi::float16 *result) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDotEx(handle,
                                                         n,
                                                         x,
                                                         CUDA_R_16F,
                                                         incx,
                                                         y,
                                                         CUDA_R_16F,
                                                         incy,
                                                         result,
                                                         CUDA_R_16F,
                                                         CUDA_R_32F));
  }
};

template <>
struct CUBlas<phi::bfloat16> {
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      cudaDataType_t Atype,
                      int lda,
                      const void *B,
                      cudaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      cudaDataType_t Ctype,
                      int ldc,
                      cudaDataType_t computeType) {
    detail::GemmEx(dev_ctx,
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
                   computeType);
  }

  static void GEMM_EX_64(phi::GPUContext *dev_ctx,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         const void *alpha,
                         const void *A,
                         cudaDataType_t Atype,
                         int64_t lda,
                         const void *B,
                         cudaDataType_t Btype,
                         int64_t ldb,
                         const void *beta,
                         void *C,
                         cudaDataType_t Ctype,
                         int64_t ldc,
                         cublasComputeType_t computeType) {
    detail::GemmEx64(dev_ctx,
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
                     computeType);
  }
};

template <>
struct CUBlas<phi::complex64> {
  static void GEMV(cublasHandle_t handle,
                   cublasOperation_t transa,
                   int m,
                   int n,
                   const phi::complex64 *alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *B,
                   int ldb,
                   const phi::complex64 *beta,
                   phi::complex64 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc));
  }

  static void GEMV_64(cublasHandle_t handle,
                      cublasOperation_t transa,
                      int64_t m,
                      int64_t n,
                      const phi::complex64 *alpha,
                      const phi::complex64 *A,
                      int64_t lda,
                      const phi::complex64 *B,
                      int64_t incx,
                      const phi::complex64 *beta,
                      phi::complex64 *C,
                      int64_t incy) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemv_v2_64(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<const cuFloatComplex *>(B),
        incx,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        incy));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMV requires CUDA 12.3 or later on Linux."));
#endif
  }

  static void AXPY(cublasHandle_t handle,
                   int n,
                   const phi::complex64 *alpha,
                   const phi::complex64 *X,
                   const int incX,
                   phi::complex64 *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCaxpy(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(X),
        incX,
        reinterpret_cast<cuFloatComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const phi::complex64 *alpha,
                                 const phi::complex64 *A,
                                 int lda,
                                 long long int strideA,    // NOLINT
                                 const phi::complex64 *B,  // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const phi::complex64 *beta,
                                 phi::complex64 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "CgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  static void GEMM(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::complex64 *alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *B,
                   int ldb,
                   const phi::complex64 *beta,
                   phi::complex64 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc));
  }

  static void GEMM_64(cublasHandle_t handle,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      const phi::complex64 *alpha,
                      const phi::complex64 *A,
                      int64_t lda,
                      const phi::complex64 *B,
                      int64_t ldb,
                      const phi::complex64 *beta,
                      phi::complex64 *C,
                      int64_t ldc) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemm_v2_64(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<const cuFloatComplex *>(B),
        ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C),
        ldc));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    detail::GemmEx(args...);
  }

  template <typename... ARGS>
  static void GEMM_EX_64(ARGS... args) {
    detail::GemmEx64(args...);
  }

  static void TRSM(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t transa,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::complex64 *alpha,
                   const phi::complex64 *A,
                   int lda,
                   phi::complex64 *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A),
        lda,
        reinterpret_cast<cuFloatComplex *>(B),
        ldb));
  }

  static void TRSM_BATCH(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t transa,
                         cublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::complex64 *alpha,
                         const phi::complex64 **A,
                         int lda,
                         phi::complex64 **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex **>(A),
        lda,
        reinterpret_cast<cuFloatComplex **>(B),
        ldb,
        batch_size));
  }

  static void GETRF_BATCH(cublasHandle_t handle,
                          int n,
                          phi::complex64 **A,
                          int lda,
                          int *ipiv,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgetrfBatched(
        handle,
        n,
        reinterpret_cast<cuFloatComplex **>(A),
        lda,
        ipiv,
        info,
        batch_size));
  }

  static void GETRI_BATCH(cublasHandle_t handle,
                          int n,
                          const phi::complex64 **A,
                          int lda,
                          const int *ipiv,
                          phi::complex64 **Ainv,
                          int ldc,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgetriBatched(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex **>(A),
        lda,
        ipiv,
        reinterpret_cast<cuFloatComplex **>(Ainv),
        ldc,
        info,
        batch_size));
  }

  static void MATINV_BATCH(cublasHandle_t handle,
                           int n,
                           const phi::complex64 **A,
                           int lda,
                           phi::complex64 **Ainv,
                           int lda_inv,
                           int *info,
                           int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCmatinvBatched(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex **>(A),
        lda,
        reinterpret_cast<cuFloatComplex **>(Ainv),
        lda_inv,
        info,
        batch_size));
  }

  static void DOT(cublasHandle_t handle,
                  int n,
                  const phi::complex64 *x,
                  const int incx,
                  const phi::complex64 *y,
                  const int incy,
                  phi::complex64 *result) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCdotu_v2(
        handle,
        n,
        reinterpret_cast<const cuFloatComplex *>(x),
        incx,
        reinterpret_cast<const cuFloatComplex *>(y),
        incy,
        reinterpret_cast<cuFloatComplex *>(result)));
  }
};

template <>
struct CUBlas<phi::complex128> {
  static void GEMV(cublasHandle_t handle,
                   cublasOperation_t transa,
                   int m,
                   int n,
                   const phi::complex128 *alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *B,
                   int ldb,
                   const phi::complex128 *beta,
                   phi::complex128 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc));
  }

  static void GEMV_64(cublasHandle_t handle,
                      cublasOperation_t transa,
                      int64_t m,
                      int64_t n,
                      const phi::complex128 *alpha,
                      const phi::complex128 *A,
                      int64_t lda,
                      const phi::complex128 *B,
                      int64_t incx,
                      const phi::complex128 *beta,
                      phi::complex128 *C,
                      int64_t incy) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemv_v2_64(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<const cuDoubleComplex *>(B),
        incx,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        incy));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMV requires CUDA 12.3 or later on Linux."));
#endif
  }

  static void AXPY(cublasHandle_t handle,
                   int n,
                   const phi::complex128 *alpha,
                   const phi::complex128 *X,
                   const int incX,
                   phi::complex128 *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZaxpy(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(X),
        incX,
        reinterpret_cast<cuDoubleComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const phi::complex128 *alpha,
                                 const phi::complex128 *A,
                                 int lda,
                                 long long int strideA,     // NOLINT
                                 const phi::complex128 *B,  // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const phi::complex128 *beta,
                                 phi::complex128 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "CgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  static void GEMM(cublasHandle_t handle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::complex128 *alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *B,
                   int ldb,
                   const phi::complex128 *beta,
                   phi::complex128 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc));
  }

  static void GEMM_64(cublasHandle_t handle,
                      cublasOperation_t transa,
                      cublasOperation_t transb,
                      int64_t m,
                      int64_t n,
                      int64_t k,
                      const phi::complex128 *alpha,
                      const phi::complex128 *A,
                      int64_t lda,
                      const phi::complex128 *B,
                      int64_t ldb,
                      const phi::complex128 *beta,
                      phi::complex128 *C,
                      int64_t ldc) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemm_v2_64(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<const cuDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C),
        ldc));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    detail::GemmEx(args...);
  }

  template <typename... ARGS>
  static void GEMM_EX_64(ARGS... args) {
    detail::GemmEx64(args...);
  }

  static void TRSM(cublasHandle_t handle,
                   cublasSideMode_t side,
                   cublasFillMode_t uplo,
                   cublasOperation_t transa,
                   cublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::complex128 *alpha,
                   const phi::complex128 *A,
                   int lda,
                   phi::complex128 *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A),
        lda,
        reinterpret_cast<cuDoubleComplex *>(B),
        ldb));
  }

  static void TRSM_BATCH(cublasHandle_t handle,
                         cublasSideMode_t side,
                         cublasFillMode_t uplo,
                         cublasOperation_t transa,
                         cublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::complex128 *alpha,
                         const phi::complex128 **A,
                         int lda,
                         phi::complex128 **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex **>(A),
        lda,
        reinterpret_cast<cuDoubleComplex **>(B),
        ldb,
        batch_size));
  }

  static void GETRF_BATCH(cublasHandle_t handle,
                          int n,
                          phi::complex128 **A,
                          int lda,
                          int *ipiv,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgetrfBatched(
        handle,
        n,
        reinterpret_cast<cuDoubleComplex **>(A),
        lda,
        ipiv,
        info,
        batch_size));
  }

  static void GETRI_BATCH(cublasHandle_t handle,
                          int n,
                          const phi::complex128 **A,
                          int lda,
                          const int *ipiv,
                          phi::complex128 **Ainv,
                          int ldc,
                          int *info,
                          int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgetriBatched(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex **>(A),
        lda,
        ipiv,
        reinterpret_cast<cuDoubleComplex **>(Ainv),
        ldc,
        info,
        batch_size));
  }

  static void MATINV_BATCH(cublasHandle_t handle,
                           int n,
                           const phi::complex128 **A,
                           int lda,
                           phi::complex128 **Ainv,
                           int lda_inv,
                           int *info,
                           int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZmatinvBatched(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex **>(A),
        lda,
        reinterpret_cast<cuDoubleComplex **>(Ainv),
        lda_inv,
        info,
        batch_size));
  }

  static void DOT(cublasHandle_t handle,
                  int n,
                  const phi::complex128 *x,
                  const int incx,
                  const phi::complex128 *y,
                  const int incy,
                  phi::complex128 *result) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZdotu_v2(
        handle,
        n,
        reinterpret_cast<const cuDoubleComplex *>(x),
        incx,
        reinterpret_cast<const cuDoubleComplex *>(y),
        incy,
        reinterpret_cast<cuDoubleComplex *>(result)));
  }
};

inline void CheckGEMMNSize(int64_t N) {
  constexpr int64_t kMaxN = 1073741823;
  if (N > kMaxN) {
    PADDLE_THROW(common::errors::Unimplemented(
        "cublas GEMM does not support N > %ld. Got N = %ld. ", kMaxN, N));
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  const bool requires_64_bit_blas =
      M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE;
  if (requires_64_bit_blas) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    auto gemm_64 = [&](cublasHandle_t handle) {
      CUBlas<T>::GEMM_64(handle,
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
    };
    if constexpr (std::is_same<T, float>::value) {
      detail::CublasCallWithTF32MathMode(&dev_ctx_, gemm_64);
    } else {
      dev_ctx_.CublasCall(gemm_64);
    }
    return;
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }
#if CUDA_VERSION >= 8000
  if constexpr (std::is_same<T, float>::value) {
    detail::CublasCallWithTF32MathMode(&dev_ctx_, [&](cublasHandle_t handle) {
      CUBlas<float>::GEMM(handle,
                          cuTransB,
                          cuTransA,
                          static_cast<int>(N),
                          static_cast<int>(M),
                          static_cast<int>(K),
                          &alpha,
                          B,
                          static_cast<int>(ldb),
                          A,
                          static_cast<int>(lda),
                          &beta,
                          C,
                          static_cast<int>(N));
    });
    return;
  }
#endif  // CUDA_VERSION >= 8000
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GEMM(handle,
                    cuTransB,
                    cuTransA,
                    static_cast<int>(N),
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &alpha,
                    B,
                    static_cast<int>(ldb),
                    A,
                    static_cast<int>(lda),
                    &beta,
                    C,
                    static_cast<int>(N));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::float16 alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        phi::float16 beta,
                                        phi::float16 *C) const {
#if CUDA_VERSION >= 8000
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  // Use FP16 input/output with FP32 accumulation.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::float16>::GEMM_EX_64(&cuda_ctx,
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
                                     CUBLAS_COMPUTE_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::float16>::GEMM_EX(&cuda_ctx,
                                  cuTransB,
                                  cuTransA,
                                  static_cast<int>(N),
                                  static_cast<int>(M),
                                  static_cast<int>(K),
                                  &h_alpha,
                                  B,
                                  CUDA_R_16F,
                                  static_cast<int>(ldb),
                                  A,
                                  CUDA_R_16F,
                                  static_cast<int>(lda),
                                  &h_beta,
                                  C,
                                  CUDA_R_16F,
                                  static_cast<int>(N),
                                  CUDA_R_32F);
  }
#else
  const int m = detail::to_blas_int(M, "GEMM M");
  const int n = detail::to_blas_int(N, "GEMM N");
  const int k = detail::to_blas_int(K, "GEMM K");
  const int lda =
      detail::to_blas_int((transA == CblasNoTrans) ? K : M, "GEMM lda");
  const int ldb =
      detail::to_blas_int((transB == CblasNoTrans) ? N : K, "GEMM ldb");
  const cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::float16>::GEMM(handle,
                               cuTransB,
                               cuTransA,
                               n,
                               m,
                               k,
                               &alpha,
                               B,
                               ldb,
                               A,
                               lda,
                               &beta,
                               C,
                               n);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <typename T, typename U>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 U alpha,
                                 const T *A,
                                 const T *B,
                                 U beta,
                                 T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  T t_alpha = static_cast<T>(alpha);
  T t_beta = static_cast<T>(beta);
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  const bool requires_64_bit_blas =
      M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE;
  if (requires_64_bit_blas) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    auto gemm_64 = [&](cublasHandle_t handle) {
      CUBlas<T>::GEMM_64(handle,
                         cuTransB,
                         cuTransA,
                         N,
                         M,
                         K,
                         &t_alpha,
                         B,
                         ldb,
                         A,
                         lda,
                         &t_beta,
                         C,
                         N);
    };
    if constexpr (std::is_same<T, float>::value) {
      detail::CublasCallWithTF32MathMode(&dev_ctx_, gemm_64);
    } else {
      dev_ctx_.CublasCall(gemm_64);
    }
    return;
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

#if CUDA_VERSION >= 8000
  if constexpr (std::is_same<T, float>::value) {
    detail::CublasCallWithTF32MathMode(&dev_ctx_, [&](cublasHandle_t handle) {
      CUBlas<float>::GEMM(handle,
                          cuTransB,
                          cuTransA,
                          static_cast<int>(N),
                          static_cast<int>(M),
                          static_cast<int>(K),
                          &t_alpha,
                          B,
                          static_cast<int>(ldb),
                          A,
                          static_cast<int>(lda),
                          &t_beta,
                          C,
                          static_cast<int>(N));
    });
    return;
  }
#endif  // CUDA_VERSION >= 8000
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GEMM(handle,
                    cuTransB,
                    cuTransA,
                    static_cast<int>(N),
                    static_cast<int>(M),
                    static_cast<int>(K),
                    &t_alpha,
                    B,
                    static_cast<int>(ldb),
                    A,
                    static_cast<int>(lda),
                    &t_beta,
                    C,
                    static_cast<int>(N));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        float alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        float beta,
                                        phi::float16 *C) const {
#if CUDA_VERSION >= 8000
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = alpha;
  float h_beta = beta;

  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  // Use FP16 input/output with FP32 accumulation.
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::float16>::GEMM_EX_64(&cuda_ctx,
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
                                     CUBLAS_COMPUTE_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::float16>::GEMM_EX(&cuda_ctx,
                                  cuTransB,
                                  cuTransA,
                                  static_cast<int>(N),
                                  static_cast<int>(M),
                                  static_cast<int>(K),
                                  &h_alpha,
                                  B,
                                  CUDA_R_16F,
                                  static_cast<int>(ldb),
                                  A,
                                  CUDA_R_16F,
                                  static_cast<int>(lda),
                                  &h_beta,
                                  C,
                                  CUDA_R_16F,
                                  static_cast<int>(N),
                                  CUDA_R_32F);
  }
#else
  const int m = detail::to_blas_int(M, "GEMM M");
  const int n = detail::to_blas_int(N, "GEMM N");
  const int k = detail::to_blas_int(K, "GEMM K");
  const int lda =
      detail::to_blas_int((transA == CblasNoTrans) ? K : M, "GEMM lda");
  const int ldb =
      detail::to_blas_int((transB == CblasNoTrans) ? N : K, "GEMM ldb");
  const cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const phi::float16 h_alpha = static_cast<phi::float16>(alpha);
  const phi::float16 h_beta = static_cast<phi::float16>(beta);
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::float16>::GEMM(handle,
                               cuTransB,
                               cuTransA,
                               n,
                               m,
                               k,
                               &h_alpha,
                               B,
                               ldb,
                               A,
                               lda,
                               &h_beta,
                               C,
                               n);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        float alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        float beta,
                                        float *C) const {
#if CUDA_VERSION >= 8000
  // cuBLAS uses column-major order, so swap A/B and M/N for row-major input.
  // The int casts below are safe in the non-64-bit branch.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "The GPU compute capability for FP16 GEMM with FP32 output must be "
          ">= 53, but received %d.",
          dev_ctx_.GetComputeCapability()));

  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::float16>::GEMM_EX_64(&cuda_ctx,
                                     cuTransB,
                                     cuTransA,
                                     N,
                                     M,
                                     K,
                                     &alpha,
                                     B,
                                     CUDA_R_16F,
                                     ldb,
                                     A,
                                     CUDA_R_16F,
                                     lda,
                                     &beta,
                                     C,
                                     CUDA_R_32F,
                                     N,
                                     CUBLAS_COMPUTE_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit FP16 GEMM_EX requires CUDA 12.3 or later on Linux."));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::float16>::GEMM_EX(&cuda_ctx,
                                  cuTransB,
                                  cuTransA,
                                  static_cast<int>(N),
                                  static_cast<int>(M),
                                  static_cast<int>(K),
                                  &alpha,
                                  B,
                                  CUDA_R_16F,
                                  static_cast<int>(ldb),
                                  A,
                                  CUDA_R_16F,
                                  static_cast<int>(lda),
                                  &beta,
                                  C,
                                  CUDA_R_32F,
                                  static_cast<int>(N),
                                  CUDA_R_32F);
  }
#else
  const int m = detail::to_blas_int(M, "GEMM M");
  const int n = detail::to_blas_int(N, "GEMM N");
  const int k = detail::to_blas_int(K, "GEMM K");
  const int lda_int = detail::to_blas_int(lda, "GEMM lda");
  const int ldb_int = detail::to_blas_int(ldb, "GEMM ldb");
  const int ldc_int = detail::to_blas_int(ldc, "GEMM ldc");
  const cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<phi::float16>::GEMM(handle,
                               cuTransB,
                               cuTransA,
                               n,
                               m,
                               k,
                               &alpha,
                               B,
                               ldb_int,
                               A,
                               lda_int,
                               &beta,
                               C,
                               ldc_int);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::bfloat16 alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        phi::bfloat16 beta,
                                        phi::bfloat16 *C) const {
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::bfloat16>::GEMM_EX_64(&cuda_ctx,
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
                                      CUBLAS_COMPUTE_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::bfloat16>::GEMM_EX(&cuda_ctx,
                                   cuTransB,
                                   cuTransA,
                                   static_cast<int>(N),
                                   static_cast<int>(M),
                                   static_cast<int>(K),
                                   &h_alpha,
                                   B,
                                   CUDA_R_16BF,
                                   static_cast<int>(ldb),
                                   A,
                                   CUDA_R_16BF,
                                   static_cast<int>(lda),
                                   &h_beta,
                                   C,
                                   CUDA_R_16BF,
                                   static_cast<int>(N),
                                   CUDA_R_32F);
  }
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}

template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        float alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        float beta,
                                        float *C) const {
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention. The int casts of lda/ldb below are safe because
  // this branch is only reached when M, N, K <= INT_MAX_VALUE.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm with fp32 output requires GPU compute capability "
          ">= 80, but received %d",
          dev_ctx_.GetComputeCapability()));

  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::bfloat16>::GEMM_EX_64(&cuda_ctx,
                                      cuTransB,
                                      cuTransA,
                                      N,
                                      M,
                                      K,
                                      &alpha,
                                      B,
                                      CUDA_R_16BF,
                                      ldb,
                                      A,
                                      CUDA_R_16BF,
                                      lda,
                                      &beta,
                                      C,
                                      CUDA_R_32F,
                                      N,
                                      CUBLAS_COMPUTE_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::bfloat16>::GEMM_EX(&cuda_ctx,
                                   cuTransB,
                                   cuTransA,
                                   static_cast<int>(N),
                                   static_cast<int>(M),
                                   static_cast<int>(K),
                                   &alpha,
                                   B,
                                   CUDA_R_16BF,
                                   static_cast<int>(ldb),
                                   A,
                                   CUDA_R_16BF,
                                   static_cast<int>(lda),
                                   &beta,
                                   C,
                                   CUDA_R_32F,
                                   static_cast<int>(N),
                                   CUDA_R_32F);
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));
#endif  // CUDA_VERSION >= 11000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        float alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        float beta,
                                        phi::bfloat16 *C) const {
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = alpha;
  float h_beta = beta;

  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::bfloat16>::GEMM_EX_64(&cuda_ctx,
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
                                      CUBLAS_COMPUTE_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::bfloat16>::GEMM_EX(&cuda_ctx,
                                   cuTransB,
                                   cuTransA,
                                   static_cast<int>(N),
                                   static_cast<int>(M),
                                   static_cast<int>(K),
                                   &h_alpha,
                                   B,
                                   CUDA_R_16BF,
                                   static_cast<int>(ldb),
                                   A,
                                   CUDA_R_16BF,
                                   static_cast<int>(lda),
                                   &h_beta,
                                   C,
                                   CUDA_R_16BF,
                                   static_cast<int>(N),
                                   CUDA_R_32F);
  }
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::complex64 alpha,
                                        const phi::complex64 *A,
                                        const phi::complex64 *B,
                                        phi::complex64 beta,
                                        phi::complex64 *C) const {
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas complex64 gemm requires GPU compute capability >= 53,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<phi::complex64>::GEMM_64(handle,
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
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<phi::complex64>::GEMM(handle,
                                   cuTransB,
                                   cuTransA,
                                   static_cast<int>(N),
                                   static_cast<int>(M),
                                   static_cast<int>(K),
                                   &alpha,
                                   B,
                                   static_cast<int>(ldb),
                                   A,
                                   static_cast<int>(lda),
                                   &beta,
                                   C,
                                   static_cast<int>(N));
    });
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::complex128 alpha,
                                        const phi::complex128 *A,
                                        const phi::complex128 *B,
                                        phi::complex128 beta,
                                        phi::complex128 *C) const {
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas complex128 gemm requires GPU compute capability >= 53,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<phi::complex128>::GEMM_64(handle,
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
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<phi::complex128>::GEMM(handle,
                                    cuTransB,
                                    cuTransA,
                                    static_cast<int>(N),
                                    static_cast<int>(M),
                                    static_cast<int>(K),
                                    &alpha,
                                    B,
                                    static_cast<int>(ldb),
                                    A,
                                    static_cast<int>(lda),
                                    &beta,
                                    C,
                                    static_cast<int>(N));
    });
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(bool transA,
                                 bool transB,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 T alpha,
                                 const T *A,
                                 int64_t lda,
                                 const T *B,
                                 int64_t ldb,
                                 T beta,
                                 T *C,
                                 int64_t ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  detail::check_blas_int64(lda, "GEMM lda");
  detail::check_blas_int64(ldb, "GEMM ldb");
  detail::check_blas_int64(ldc, "GEMM ldc");
  const bool requires_64_bit_blas = M > INT_MAX_VALUE || N > INT_MAX_VALUE ||
                                    K > INT_MAX_VALUE || lda > INT_MAX_VALUE ||
                                    ldb > INT_MAX_VALUE || ldc > INT_MAX_VALUE;
  if (requires_64_bit_blas) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    auto gemm_64 = [&](cublasHandle_t handle) {
      CUBlas<T>::GEMM_64(handle,
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
    };
    if constexpr (std::is_same<T, float>::value) {
      detail::CublasCallWithTF32MathMode(&dev_ctx_, gemm_64);
    } else {
      dev_ctx_.CublasCall(gemm_64);
    }
    return;
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }
  const int m = static_cast<int>(M);
  const int n = static_cast<int>(N);
  const int k = static_cast<int>(K);
  const int lda_int = static_cast<int>(lda);
  const int ldb_int = static_cast<int>(ldb);
  const int ldc_int = static_cast<int>(ldc);

#if CUDA_VERSION >= 8000
  if constexpr (std::is_same<T, float>::value) {
    detail::CublasCallWithTF32MathMode(&dev_ctx_, [&](cublasHandle_t handle) {
      CUBlas<float>::GEMM(handle,
                          cuTransB,
                          cuTransA,
                          n,
                          m,
                          k,
                          &alpha,
                          B,
                          ldb_int,
                          A,
                          lda_int,
                          &beta,
                          C,
                          ldc_int);
    });
    return;
  }
#endif  // CUDA_VERSION >= 8000

  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GEMM(handle,
                    cuTransB,
                    cuTransA,
                    n,
                    m,
                    k,
                    &alpha,
                    B,
                    ldb_int,
                    A,
                    lda_int,
                    &beta,
                    C,
                    ldc_int);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::float16 alpha,
                                        const phi::float16 *A,
                                        int64_t lda,
                                        const phi::float16 *B,
                                        int64_t ldb,
                                        phi::float16 beta,
                                        phi::float16 *C,
                                        int64_t ldc) const {
#if CUDA_VERSION >= 8000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  detail::check_blas_int64(lda, "GEMM lda");
  detail::check_blas_int64(ldb, "GEMM ldb");
  detail::check_blas_int64(ldc, "GEMM ldc");
  const bool requires_64_bit_blas = M > INT_MAX_VALUE || N > INT_MAX_VALUE ||
                                    K > INT_MAX_VALUE || lda > INT_MAX_VALUE ||
                                    ldb > INT_MAX_VALUE || ldc > INT_MAX_VALUE;

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);

  if (requires_64_bit_blas) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::float16>::GEMM_EX_64(&cuda_ctx,
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
                                     ldc,
                                     CUBLAS_COMPUTE_32F);
    return;
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

  const int m = static_cast<int>(M);
  const int n = static_cast<int>(N);
  const int k = static_cast<int>(K);
  const int lda_int = static_cast<int>(lda);
  const int ldb_int = static_cast<int>(ldb);
  const int ldc_int = static_cast<int>(ldc);
  CheckGEMMNSize(n);
  CUBlas<phi::float16>::GEMM_EX(&cuda_ctx,
                                cuTransB,
                                cuTransA,
                                n,
                                m,
                                k,
                                &h_alpha,
                                B,
                                CUDA_R_16F,
                                ldb_int,
                                A,
                                CUDA_R_16F,
                                lda_int,
                                &h_beta,
                                C,
                                CUDA_R_16F,
                                ldc_int,
                                CUDA_R_32F);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "FP16 GEMM_EX requires CUDA 8.0 or later."));
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::bfloat16 alpha,
                                        const phi::bfloat16 *A,
                                        int64_t lda,
                                        const phi::bfloat16 *B,
                                        int64_t ldb,
                                        phi::bfloat16 beta,
                                        phi::bfloat16 *C,
                                        int64_t ldc) const {
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  detail::check_blas_int64(M, "GEMM M");
  detail::check_blas_int64(N, "GEMM N");
  detail::check_blas_int64(K, "GEMM K");
  detail::check_blas_int64(lda, "GEMM lda");
  detail::check_blas_int64(ldb, "GEMM ldb");
  detail::check_blas_int64(ldc, "GEMM ldc");
  const bool requires_64_bit_blas = M > INT_MAX_VALUE || N > INT_MAX_VALUE ||
                                    K > INT_MAX_VALUE || lda > INT_MAX_VALUE ||
                                    ldb > INT_MAX_VALUE || ldc > INT_MAX_VALUE;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);

  if (requires_64_bit_blas) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    CUBlas<phi::bfloat16>::GEMM_EX_64(&cuda_ctx,
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
                                      ldc,
                                      CUBLAS_COMPUTE_32F);
    return;
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMM requires CUDA 12.3 or later on Linux."));
#endif
  }

  const int m = static_cast<int>(M);
  const int n = static_cast<int>(N);
  const int k = static_cast<int>(K);
  const int lda_int = static_cast<int>(lda);
  const int ldb_int = static_cast<int>(ldb);
  const int ldc_int = static_cast<int>(ldc);
  CheckGEMMNSize(n);
  CUBlas<phi::bfloat16>::GEMM_EX(&cuda_ctx,
                                 cuTransB,
                                 cuTransA,
                                 n,
                                 m,
                                 k,
                                 &h_alpha,
                                 B,
                                 CUDA_R_16BF,
                                 ldb_int,
                                 A,
                                 CUDA_R_16BF,
                                 lda_int,
                                 &h_beta,
                                 C,
                                 CUDA_R_16BF,
                                 ldc_int,
                                 CUDA_R_32F);
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <typename T>
void Blas<phi::GPUContext>::AXPY(int64_t n, T alpha, const T *x, T *y) const {
  if (n <= 0) {
    return;
  }
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    int64_t offset = 0;
    while (offset < n) {
      const int chunk_size = static_cast<int>(
          n - offset > INT_MAX_VALUE ? INT_MAX_VALUE : n - offset);
      CUBlas<T>::AXPY(handle, chunk_size, &alpha, x + offset, 1, y + offset, 1);
      offset += chunk_size;
    }
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::CUDOT(int64_t n,
                                  const T *x,
                                  int64_t incx,
                                  const T *y,
                                  int64_t incy,
                                  T *result) const {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  const int n_int = detail::to_blas_int(n, "CUDOT n");
  const int incx_int = detail::to_blas_int(incx, "CUDOT incx");
  const int incy_int = detail::to_blas_int(incy, "CUDOT incy");
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::DOT(handle, n_int, x, incx_int, y, incy_int, result);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::CUDOT(int64_t n,
                                         const phi::bfloat16 *x,
                                         int64_t incx,
                                         const phi::bfloat16 *y,
                                         int64_t incy,
                                         phi::bfloat16 *result) const {
  if (n == 1) {
    incx = 1;
    incy = 1;
  }
  const int n_int = detail::to_blas_int(n, "CUDOT n");
  const int incx_int = detail::to_blas_int(incx, "CUDOT incx");
  const int incy_int = detail::to_blas_int(incy, "CUDOT incy");
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDotEx(handle,
                                                         n_int,
                                                         x,
                                                         CUDA_R_16BF,
                                                         incx_int,
                                                         y,
                                                         CUDA_R_16BF,
                                                         incy_int,
                                                         result,
                                                         CUDA_R_16BF,
                                                         CUDA_R_32F));
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                 int64_t M,
                                 int64_t N,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  detail::check_blas_int64(M, "GEMV M");
  detail::check_blas_int64(N, "GEMV N");
  cublasOperation_t cuTransA = !trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMV_64(
          handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
    });
    return;
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "64-bit cuBLAS GEMV requires CUDA 12.3 or later on Linux."));
#endif
  }
  const int m = static_cast<int>(M);
  const int n = static_cast<int>(N);
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GEMV(handle, cuTransA, n, m, &alpha, A, n, B, 1, &beta, C, 1);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int64_t M,
                                        int64_t N,
                                        phi::float16 alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        phi::float16 beta,
                                        phi::float16 *C) const {
  // cuBLAS has no FP16 GEMV API, so express the operation as GEMM.
  if (trans_a) {
    this->template GEMM<phi::float16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::float16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int64_t M,
                                        int64_t N,
                                        phi::bfloat16 alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        phi::bfloat16 beta,
                                        phi::bfloat16 *C) const {
  // cuBLAS has no BF16 GEMV API, so express the operation as GEMM.
  if (trans_a) {
    this->template GEMM<phi::bfloat16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::bfloat16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;
#if CUDA_VERSION >= 9010
  if ((std::is_same<T, float>::value) || std::is_same<T, phi::float16>::value) {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
    bool use_tensor_op_math = std::is_same<T, float>::value
                                  ? FLAGS_cublas_allow_tf32
                                  : dev_ctx_.tensor_core_available();
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
        std::is_same<T, phi::float16>::value) {
      a = static_cast<void *>(&alpha);
      b = static_cast<void *>(&beta);
#if CUDA_VERSION >= 11000
      compute_type = CUBLAS_COMPUTE_16F;
#else
      compute_type = CUDA_R_16F;
#endif
    }
    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
      detail::CublasCallForGemmEx<T>(&dev_ctx_, [&](cublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cublasGemmStridedBatchedEx_64(handle,
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
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "cublasGemmStridedBatchedEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
    } else {
      detail::CublasCallForGemmEx<T>(&dev_ctx_, [&](cublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmStridedBatchedEx(
            handle,
            cuTransB,
            cuTransA,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            a,
            B,
            fp,
            static_cast<int>(ldb),
            strideB,
            A,
            fp,
            static_cast<int>(lda),
            strideA,
            b,
            C,
            fp,
            static_cast<int>(ldc),
            strideC,
            static_cast<int>(batchCount),
            compute_type,
            algo));
      });
    }
  } else {
#endif  // CUDA_VERSION >= 9010
    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP) && !defined(_WIN32)
      if (N == 1 && ldc >= std::max<int64_t>(1, M) && !FLAGS_use_legacy_gemm) {
        // No transpose result in these case, align with torch's behaviour.
        // TODO(Pan Zhaowu): Integrate proper stride support for arbitrary input
        // tensor.
        CUBlas<T>::GEMM_STRIDED_BATCH(
            handle,
            (cuTransA == CUBLAS_OP_T) ? CUBLAS_OP_N : CUBLAS_OP_T,
            (cuTransB == CUBLAS_OP_T) ? CUBLAS_OP_N : CUBLAS_OP_T,
            static_cast<int>(M),
            static_cast<int>(N),
            static_cast<int>(K),
            &alpha,
            A,
            static_cast<int>(lda),
            strideA,
            B,
            static_cast<int>(ldb),
            strideB,
            &beta,
            C,
            static_cast<int>(ldc),
            strideC,
            static_cast<int>(batchCount));
      } else  // NOLINT
#endif
      {
        CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                      cuTransB,
                                      cuTransA,
                                      static_cast<int>(N),
                                      static_cast<int>(M),
                                      static_cast<int>(K),
                                      &alpha,
                                      B,
                                      static_cast<int>(ldb),
                                      strideB,
                                      A,
                                      static_cast<int>(lda),
                                      strideA,
                                      &beta,
                                      C,
                                      static_cast<int>(ldc),
                                      strideC,
                                      static_cast<int>(batchCount));
      }
    });

#if CUDA_VERSION >= 9010
  }
#endif  // CUDA_VERSION >= 9010
}

template <>
template <typename T, typename U>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
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
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;
#if CUDA_VERSION >= 9010
  if ((std::is_same<T, float>::value) || std::is_same<T, phi::float16>::value) {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
    bool use_tensor_op_math = std::is_same<T, float>::value
                                  ? FLAGS_cublas_allow_tf32
                                  : dev_ctx_.tensor_core_available();
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
        std::is_same<T, phi::float16>::value) {
      a = static_cast<void *>(&alpha);
      b = static_cast<void *>(&beta);
#if CUDA_VERSION >= 11000
      compute_type = CUBLAS_COMPUTE_16F;
#else
      compute_type = CUDA_R_16F;
#endif
    }

    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
        batchCount > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
      detail::CublasCallForGemmEx<T>(&dev_ctx_, [&](cublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cublasGemmStridedBatchedEx_64(handle,
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
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "cublasGemmStridedBatchedEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
    } else {
      detail::CublasCallForGemmEx<T>(&dev_ctx_, [&](cublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasGemmStridedBatchedEx(
            handle,
            cuTransB,
            cuTransA,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            a,
            B,
            fp,
            static_cast<int>(ldb),
            strideB,
            A,
            fp,
            static_cast<int>(lda),
            strideA,
            b,
            C,
            fp,
            static_cast<int>(ldc),
            strideC,
            static_cast<int>(batchCount),
            compute_type,
            algo));
      });
    }
  } else {
#endif  // CUDA_VERSION >= 9010
    T h_alpha = static_cast<T>(alpha);
    T h_beta = static_cast<T>(beta);

    dev_ctx_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                    cuTransB,
                                    cuTransA,
                                    static_cast<int>(N),
                                    static_cast<int>(M),
                                    static_cast<int>(K),
                                    &h_alpha,
                                    B,
                                    static_cast<int>(ldb),
                                    strideB,
                                    A,
                                    static_cast<int>(lda),
                                    strideA,
                                    &h_beta,
                                    C,
                                    static_cast<int>(ldc),
                                    strideC,
                                    static_cast<int>(batchCount));
    });

#if CUDA_VERSION >= 9010
  }
#endif  // CUDA_VERSION >= 9010
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int64_t M,
                                               int64_t N,
                                               int64_t K,
                                               phi::bfloat16 alpha,
                                               const phi::bfloat16 *A,
                                               const phi::bfloat16 *B,
                                               phi::bfloat16 beta,
                                               phi::bfloat16 *C,
                                               int64_t batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;

  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
      batchCount > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx_64(handle,
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
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmStridedBatchedEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   static_cast<int>(N),
                                                   static_cast<int>(M),
                                                   static_cast<int>(K),
                                                   &h_alpha,
                                                   B,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(ldb),
                                                   strideB,
                                                   A,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(lda),
                                                   strideA,
                                                   &h_beta,
                                                   C,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(ldc),
                                                   strideC,
                                                   static_cast<int>(batchCount),
                                                   CUBLAS_COMPUTE_32F,
                                                   algo));
    });
  }
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmStridedBatchedEx with bfloat16 is not supported on cuda <= "
      "11"));
#endif  // CUDA_VERSION >= 11000
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int64_t M,
                                               int64_t N,
                                               int64_t K,
                                               float alpha,
                                               const phi::bfloat16 *A,
                                               const phi::bfloat16 *B,
                                               float beta,
                                               phi::bfloat16 *C,
                                               int64_t batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
#if CUDA_VERSION >= 11000
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  float h_alpha = alpha;
  float h_beta = beta;

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
      batchCount > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx_64(handle,
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
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmStridedBatchedEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030
  } else {
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   static_cast<int>(N),
                                                   static_cast<int>(M),
                                                   static_cast<int>(K),
                                                   &h_alpha,
                                                   B,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(ldb),
                                                   strideB,
                                                   A,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(lda),
                                                   strideA,
                                                   &h_beta,
                                                   C,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(ldc),
                                                   strideC,
                                                   static_cast<int>(batchCount),
                                                   CUBLAS_COMPUTE_32F,
                                                   algo));
    });
  }
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmStridedBatchedEx with bfloat16 is not supported on cuda <= "
      "11"));
#endif  // CUDA_VERSION >= 11000
}

template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int64_t M,
                                               int64_t N,
                                               int64_t K,
                                               float alpha,
                                               const phi::float16 *A,
                                               const phi::float16 *B,
                                               float beta,
                                               float *C,
                                               int64_t batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
#if CUDA_VERSION >= 8000
  // cuBLAS uses column-major order, so A and B are swapped below.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 batched gemm with fp32 output requires GPU compute "
          "capability >= 53, but received %d",
          dev_ctx_.GetComputeCapability()));

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
#if CUDA_VERSION >= 9000
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
#endif
#if CUDA_VERSION >= 11000
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
      batchCount > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx_64(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &alpha,
                                                      B,
                                                      CUDA_R_16F,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      CUDA_R_16F,
                                                      lda,
                                                      strideA,
                                                      &beta,
                                                      C,
                                                      CUDA_R_32F,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      compute_type,
                                                      algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmStridedBatchedEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030 && defined(__linux__)
  } else {
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   static_cast<int>(N),
                                                   static_cast<int>(M),
                                                   static_cast<int>(K),
                                                   &alpha,
                                                   B,
                                                   CUDA_R_16F,
                                                   static_cast<int>(ldb),
                                                   strideB,
                                                   A,
                                                   CUDA_R_16F,
                                                   static_cast<int>(lda),
                                                   strideA,
                                                   &beta,
                                                   C,
                                                   CUDA_R_32F,
                                                   static_cast<int>(ldc),
                                                   strideC,
                                                   static_cast<int>(batchCount),
                                                   compute_type,
                                                   algo));
    });
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmStridedBatchedEx with float16 is not supported on cuda <= "
      "7.5"));
#endif  // CUDA_VERSION >= 8000
}

template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int64_t M,
                                               int64_t N,
                                               int64_t K,
                                               float alpha,
                                               const phi::bfloat16 *A,
                                               const phi::bfloat16 *B,
                                               float beta,
                                               float *C,
                                               int64_t batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
#if CUDA_VERSION >= 11000
  // cuBLAS uses column-major order, so A and B are swapped below.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 batched gemm with fp32 output requires GPU compute "
          "capability >= 80, but received %d",
          dev_ctx_.GetComputeCapability()));

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
      batchCount > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__)
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx_64(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &alpha,
                                                      B,
                                                      CUDA_R_16BF,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      CUDA_R_16BF,
                                                      lda,
                                                      strideA,
                                                      &beta,
                                                      C,
                                                      CUDA_R_32F,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      CUBLAS_COMPUTE_32F,
                                                      algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "cublasGemmStridedBatchedEx_64 is not supported on cuda < 12.3"));
#endif  // CUDA_VERSION >= 12030 && defined(__linux__)
  } else {
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   static_cast<int>(N),
                                                   static_cast<int>(M),
                                                   static_cast<int>(K),
                                                   &alpha,
                                                   B,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(ldb),
                                                   strideB,
                                                   A,
                                                   CUDA_R_16BF,
                                                   static_cast<int>(lda),
                                                   strideA,
                                                   &beta,
                                                   C,
                                                   CUDA_R_32F,
                                                   static_cast<int>(ldc),
                                                   strideC,
                                                   static_cast<int>(batchCount),
                                                   CUBLAS_COMPUTE_32F,
                                                   algo));
    });
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmStridedBatchedEx with bfloat16 is not supported on cuda < "
      "11"));
#endif  // CUDA_VERSION >= 11000
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

#if defined(__NVCC__)
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  thrust::device_vector<const double *> A_ptr(A, A + batchCount);
  thrust::device_vector<const double *> B_ptr(B, B + batchCount);
  thrust::device_vector<double *> C_ptr(C, C + batchCount);

  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
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
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  thrust::device_vector<const float *> A_ptr(A, A + batchCount);
  thrust::device_vector<const float *> B_ptr(B, B + batchCount);
  thrust::device_vector<float *> C_ptr(C, C + batchCount);

  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
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
                                               phi::float16 alpha,
                                               const phi::float16 **A,
                                               const phi::float16 **B,
                                               phi::float16 beta,
                                               phi::float16 **C,
                                               int batchCount) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      53,
      common::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));
  float f_alpha = static_cast<float>(alpha);
  float f_beta = static_cast<float>(beta);
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  CUBlas<phi::float16>::GEMM_BATCH(&cuda_ctx,
                                   cuTransB,
                                   cuTransA,
                                   N,
                                   M,
                                   K,
                                   &f_alpha,
                                   B,
                                   CUDA_R_16F,
                                   ldb,
                                   A,
                                   CUDA_R_16F,
                                   lda,
                                   &f_beta,
                                   C,
                                   CUDA_R_16F,
                                   ldc,
                                   batchCount,
                                   CUDA_R_32F);
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::bfloat16 alpha,
                                               const phi::bfloat16 **A,
                                               const phi::bfloat16 **B,
                                               phi::bfloat16 beta,
                                               phi::bfloat16 **C,
                                               int batchCount) const {
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

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      80,
      common::errors::InvalidArgument(
          "cublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float f_alpha = static_cast<float>(alpha);
  float f_beta = static_cast<float>(beta);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  thrust::device_vector<const void *> A_ptr(A, A + batchCount);
  thrust::device_vector<const void *> B_ptr(B, B + batchCount);
  thrust::device_vector<void *> C_ptr(C, C + batchCount);
  dev_ctx_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cublasGemmBatchedEx(handle,
                                          cuTransB,
                                          cuTransA,
                                          N,
                                          M,
                                          K,
                                          &f_alpha,
                                          B_ptr.data().get(),
                                          CUDA_R_16BF,
                                          ldb,
                                          A_ptr.data().get(),
                                          CUDA_R_16BF,
                                          lda,
                                          &f_beta,
                                          C_ptr.data().get(),
                                          CUDA_R_16BF,
                                          ldc,
                                          batchCount,
                                          CUDA_R_32F,
                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "cublasGemmBatchedEx with bfloat16 is not supported on cuda <= 11"));

#endif  // CUDA_VERSION >= 11000
}
#endif

template <>
template <typename T>
void Blas<phi::GPUContext>::TRSM(CBLAS_SIDE side,
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
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  cublasSideMode_t cuSide =
      (side == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasFillMode_t cuUplo =
      (uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuDiag =
      (diag == CblasUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::TRSM(handle,
                    cuSide,
                    cuUplo,
                    cuTransA,
                    cuDiag,
                    n,
                    m,
                    &alpha,
                    A,
                    lda_int,
                    B,
                    ldb_int);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRF(
    int64_t n, T **a, int *ipiv, int *info, int64_t batch_size) const {
  const int n_int = detail::to_blas_int(n, "BatchedGETRF n");
  const int batch_size_int =
      detail::to_blas_int(batch_size, "BatchedGETRF batch_size");
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GETRF_BATCH(handle, n_int, a, n_int, ipiv, info, batch_size_int);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRI(int64_t n,
                                         const T **a,
                                         const int *ipiv,
                                         T **a_inv,
                                         int *info,
                                         int64_t batch_size) const {
  const int n_int = detail::to_blas_int(n, "BatchedGETRI n");
  const int batch_size_int =
      detail::to_blas_int(batch_size, "BatchedGETRI batch_size");
  PADDLE_ENFORCE_NE(
      a_inv,
      a,
      common::errors::InvalidArgument(
          "cuBLAS function 'cublas<S/D>getrfBatched' cannot be executed "
          "in-place. The memory space of output matrix (address: %p) cannot "
          "overlap memory space of input matrix (address: %p).",
          a_inv,
          a));
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GETRI_BATCH(
        handle, n_int, a, n_int, ipiv, a_inv, n_int, info, batch_size_int);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedMatInv(
    int64_t n, const T **a, T **a_inv, int *info, int64_t batch_size) const {
  const int n_int = detail::to_blas_int(n, "BatchedMatInv n");
  const int batch_size_int =
      detail::to_blas_int(batch_size, "BatchedMatInv batch_size");
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::MATINV_BATCH(
        handle, n_int, a, n_int, a_inv, n_int, info, batch_size_int);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRS(CBLAS_TRANSPOSE trans,
                                         int64_t n,
                                         int64_t nrhs,
                                         const T **a,
                                         int64_t lda,
                                         int *ipiv,
                                         T **b,
                                         int64_t ldb,
                                         int *info,
                                         int64_t batch_size) const {
  const int n_int = detail::to_blas_int(n, "BatchedGETRS n");
  const int nrhs_int = detail::to_blas_int(nrhs, "BatchedGETRS nrhs");
  const int lda_int = detail::to_blas_int(lda, "BatchedGETRS lda");
  const int ldb_int = detail::to_blas_int(ldb, "BatchedGETRS ldb");
  const int batch_size_int =
      detail::to_blas_int(batch_size, "BatchedGETRS batch_size");
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTrans =
      (trans == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GETRS_BATCH(handle,
                           cuTrans,
                           n_int,
                           nrhs_int,
                           a,
                           lda_int,
                           ipiv,
                           b,
                           ldb_int,
                           info,
                           batch_size_int);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedTRSM(CBLAS_SIDE side,
                                        CBLAS_UPLO uplo,
                                        CBLAS_TRANSPOSE transA,
                                        CBLAS_DIAG diag,
                                        int64_t M,
                                        int64_t N,
                                        T alpha,
                                        const T **A,
                                        int64_t lda,
                                        T **B,
                                        int64_t ldb,
                                        int64_t batch_size) const {
  const int m = detail::to_blas_int(M, "BatchedTRSM M");
  const int n = detail::to_blas_int(N, "BatchedTRSM N");
  const int lda_int = detail::to_blas_int(lda, "BatchedTRSM lda");
  const int ldb_int = detail::to_blas_int(ldb, "BatchedTRSM ldb");
  const int batch_size_int =
      detail::to_blas_int(batch_size, "BatchedTRSM batch_size");
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  cublasSideMode_t cuSide =
      (side == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasFillMode_t cuUplo =
      (uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuDiag =
      (diag == CblasUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  dev_ctx_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::TRSM_BATCH(handle,
                          cuSide,
                          cuUplo,
                          cuTransA,
                          cuDiag,
                          n,
                          m,
                          &alpha,
                          A,
                          lda_int,
                          B,
                          ldb_int,
                          batch_size_int);
  });
}

}  // namespace funcs
}  // namespace phi
