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

#if defined(PADDLE_WITH_CUDA)

#include <ATen/cuda/CUDABlas.h>

#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/core/enforce.h"

namespace at::cuda::blas {

namespace {

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

}  // namespace

/* ───────────── gemm<double> ───────────── */
template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
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
}

/* ───────────── gemm<float> ───────────── */
template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
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
}

/* ───────────── gemm<c10::complex<double>> ───────────── */
template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
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
}

/* ───────────── gemm<c10::complex<float>> ───────────── */
template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = to_cublas_op(transa);
  cublasOperation_t opb = to_cublas_op(transb);
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
}

}  // namespace at::cuda::blas

#endif  // PADDLE_WITH_CUDA
