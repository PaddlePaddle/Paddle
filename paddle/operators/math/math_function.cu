/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/math_function.h"


namespace paddle {
namespace operators {
namespace math {

template <>
void gemm<platform::GPUPlace float>(const CBLAS_TRANSPOSE transA,
                                    const CBLAS_TRANSPOSE transB,
                                    const int M,
                                    const int N,
                                    const int K,
                                    const float alpha,
                                    const float* A,
                                    const int lda,
                                    const float* B,
                                    const int ldb,
                                    const float beta,
                                    float* C,
                                    const int ldc,
                                    const platform::DeviceContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
                     
  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<const platform::CUDADeviceContext*>(context)->
        cublas_handle(),
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
      ldc));
}

template <>
void gemm<platform::GPUPlace, double>(const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB,
                                      const int M,
                                      const int N,
                                      const int K,
                                      const double alpha,
                                      const double* A,
                                      const int lda,
                                      const double* B,
                                      const int ldb,
                                      const double beta,
                                      double* C,
                                      const int ldc,
                                      const platform::DeviceContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<const platform::CUDADeviceContext*>(context)->
        cublas_handle(),
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
      ldc));
}


template <>
void axpy<platform::GPUPlace, float>(const int n, 
                                     const float alpha,
                                     const float* x,
                                     float* y,
                                     const platform::DeviceContext* context) {
  CUBLAS_ENFORCE(platform::dynload::cublasSaxpy(
    reinterpret_cast<const platform::CUDADeviceContext*>(context)->
      cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void axpy<platform::GPUPlace, double>(const int n,
                                      const double alpha,
                                      const double* x,
                                      double* y,
                                      const platform::DeviceContext* context) {
  CUBLAS_ENFORCE(platform::dynload::cublasDaxpy(
    reinterpret_cast<const platform::CUDADeviceContext*>(context)->
      cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
float dotProduct<platform::GPUPlace, float>(const int n,
                                            const float* x,
                                            const float* y,
                                            const platform::DeviceContext* context) {
  CUBLAS_ENFORCE(platform::dynload::cublasSdot(
    reinterpret_cast<const platform::CUDADeviceContext*>(context)->
      cublas_handle(), n, a, 1, b, 1, &result));
}

template <>
double dotProduct<platform::GPUPlace, double>(const int n,
                                              const double* x,
                                              const double* y,
                                              const platform::DeviceContext* context) {
  CUBLAS_ENFORCE(platform::dynload::cublasDdot(
    reinterpret_cast<const platform::CUDADeviceContext*>(context)->
      cublas_handle(), n, a, 1, b, 1, &result));
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
