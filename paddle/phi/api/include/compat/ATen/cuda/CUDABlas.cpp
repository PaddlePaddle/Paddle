// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <ATen/cuda/CUDABlas.h>

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/hipblas.h"
#else
#include "paddle/phi/backends/dynload/cublas.h"
#endif
#include "paddle/phi/core/enforce.h"

namespace at::cuda::blas {

namespace {

#ifdef PADDLE_WITH_HIP
using cublasHandle_t = hipblasHandle_t;
using cublasOperation_t = hipblasOperation_t;
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_C HIPBLAS_OP_C
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT

inline cublasOperation_t to_cublas_op(char trans) {
  switch (trans) {
    case 'T':
    case 't':
      return HIPBLAS_OP_T;
    case 'N':
    case 'n':
      return HIPBLAS_OP_N;
    case 'C':
    case 'c':
      return HIPBLAS_OP_C;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "at::cuda::blas::gemm: invalid transpose character '%c'", trans));
  }
}
#else
inline cublasOperation_t to_cublas_op(char trans) {
  switch (trans) {
    case 'T':
    case 't':
      return CUBLAS_OP_T;
    case 'N':
    case 'n':
      return CUBLAS_OP_N;
    case 'C':
    case 'c':
      return CUBLAS_OP_C;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "at::cuda::blas::gemm: invalid transpose character '%c'", trans));
  }
}
#endif

}  // namespace

/* ───────────── gemm<double> ───────────── */
template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipblasDgemm(handle,
                                                        opa,
                                                        opb,
                                                        static_cast<int>(m),
                                                        static_cast<int>(n),
                                                        static_cast<int>(k),
                                                        &alpha,
                                                        a,
                                                        static_cast<int>(lda),
                                                        b,
                                                        static_cast<int>(ldb),
                                                        &beta,
                                                        c,
                                                        static_cast<int>(ldc)));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasDgemm(handle,
                                                       opa,
                                                       opb,
                                                       static_cast<int>(m),
                                                       static_cast<int>(n),
                                                       static_cast<int>(k),
                                                       &alpha,
                                                       a,
                                                       static_cast<int>(lda),
                                                       b,
                                                       static_cast<int>(ldb),
                                                       &beta,
                                                       c,
                                                       static_cast<int>(ldc)));
#endif
}

/* ───────────── gemm<float> ───────────── */
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipblasSgemm(handle,
                                                        opa,
                                                        opb,
                                                        static_cast<int>(m),
                                                        static_cast<int>(n),
                                                        static_cast<int>(k),
                                                        &alpha,
                                                        a,
                                                        static_cast<int>(lda),
                                                        b,
                                                        static_cast<int>(ldb),
                                                        &beta,
                                                        c,
                                                        static_cast<int>(ldc)));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasSgemm(handle,
                                                       opa,
                                                       opb,
                                                       static_cast<int>(m),
                                                       static_cast<int>(n),
                                                       static_cast<int>(k),
                                                       &alpha,
                                                       a,
                                                       static_cast<int>(lda),
                                                       b,
                                                       static_cast<int>(ldb),
                                                       &beta,
                                                       c,
                                                       static_cast<int>(ldc)));
#endif
}

/* ───────────── gemm<c10::complex<double>> ───────────── */
template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipblasZgemm(
      handle,
      opa,
      opb,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<const hipDoubleComplex *>(&alpha),
      reinterpret_cast<const hipDoubleComplex *>(a),
      static_cast<int>(lda),
      reinterpret_cast<const hipDoubleComplex *>(b),
      static_cast<int>(ldb),
      reinterpret_cast<const hipDoubleComplex *>(&beta),
      reinterpret_cast<hipDoubleComplex *>(c),
      static_cast<int>(ldc)));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasZgemm(
      handle,
      opa,
      opb,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<const cuDoubleComplex *>(&alpha),
      reinterpret_cast<const cuDoubleComplex *>(a),
      static_cast<int>(lda),
      reinterpret_cast<const cuDoubleComplex *>(b),
      static_cast<int>(ldb),
      reinterpret_cast<const cuDoubleComplex *>(&beta),
      reinterpret_cast<cuDoubleComplex *>(c),
      static_cast<int>(ldc)));
#endif
}

/* ───────────── gemm<c10::complex<float>> ───────────── */
template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipblasCgemm(
      handle,
      opa,
      opb,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<const hipFloatComplex *>(&alpha),
      reinterpret_cast<const hipFloatComplex *>(a),
      static_cast<int>(lda),
      reinterpret_cast<const hipFloatComplex *>(b),
      static_cast<int>(ldb),
      reinterpret_cast<const hipFloatComplex *>(&beta),
      reinterpret_cast<hipFloatComplex *>(c),
      static_cast<int>(ldc)));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cublasCgemm(
      handle,
      opa,
      opb,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      reinterpret_cast<const cuFloatComplex *>(&alpha),
      reinterpret_cast<const cuFloatComplex *>(a),
      static_cast<int>(lda),
      reinterpret_cast<const cuFloatComplex *>(b),
      static_cast<int>(ldb),
      reinterpret_cast<const cuFloatComplex *>(&beta),
      reinterpret_cast<cuFloatComplex *>(c),
      static_cast<int>(ldc)));
#endif
}

/* ───────────── gemm<at::Half> ───────────── */
template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);

  // Use cublasGemmEx with FP32 compute for Half inputs
  float alpha_f = alpha;
  float beta_f = beta;
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipblasGemmEx(handle,
                                                         opa,
                                                         opb,
                                                         static_cast<int>(m),
                                                         static_cast<int>(n),
                                                         static_cast<int>(k),
                                                         &alpha_f,
                                                         a,
                                                         HIP_R_16F,
                                                         static_cast<int>(lda),
                                                         b,
                                                         HIP_R_16F,
                                                         static_cast<int>(ldb),
                                                         &beta_f,
                                                         c,
                                                         HIP_R_16F,
                                                         static_cast<int>(ldc),
                                                         HIP_R_32F,
                                                         HIPBLAS_GEMM_DEFAULT));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cublasGemmEx(handle,
                                 opa,
                                 opb,
                                 static_cast<int>(m),
                                 static_cast<int>(n),
                                 static_cast<int>(k),
                                 &alpha_f,
                                 a,
                                 CUDA_R_16F,
                                 static_cast<int>(lda),
                                 b,
                                 CUDA_R_16F,
                                 static_cast<int>(ldb),
                                 &beta_f,
                                 c,
                                 CUDA_R_16F,
                                 static_cast<int>(ldc),
                                 CUDA_R_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

/* ───────────── gemm<at::BFloat16> ───────────── */
template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);

  // Use cublasGemmEx with FP32 compute for BFloat16 inputs
  float alpha_f = alpha;
  float beta_f = beta;
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipblasGemmEx(handle,
                                                         opa,
                                                         opb,
                                                         static_cast<int>(m),
                                                         static_cast<int>(n),
                                                         static_cast<int>(k),
                                                         &alpha_f,
                                                         a,
                                                         HIP_R_16BF,
                                                         static_cast<int>(lda),
                                                         b,
                                                         HIP_R_16BF,
                                                         static_cast<int>(ldb),
                                                         &beta_f,
                                                         c,
                                                         HIP_R_16BF,
                                                         static_cast<int>(ldc),
                                                         HIP_R_32F,
                                                         HIPBLAS_GEMM_DEFAULT));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cublasGemmEx(handle,
                                 opa,
                                 opb,
                                 static_cast<int>(m),
                                 static_cast<int>(n),
                                 static_cast<int>(k),
                                 &alpha_f,
                                 a,
                                 CUDA_R_16BF,
                                 static_cast<int>(lda),
                                 b,
                                 CUDA_R_16BF,
                                 static_cast<int>(ldb),
                                 &beta_f,
                                 c,
                                 CUDA_R_16BF,
                                 static_cast<int>(ldc),
                                 CUDA_R_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
}

}  // namespace at::cuda::blas

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
