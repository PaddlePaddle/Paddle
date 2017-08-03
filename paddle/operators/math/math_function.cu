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
void gemm<platform::GPUPlace, float>(const CBLAS_TRANSPOSE transA,
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
                                     platform::DeviceContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
                     
  PADDLE_ENFORCE(platform::dynload::cublasSgemm(
      reinterpret_cast<platform::CUDADeviceContext*>(context)->
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
                                      platform::DeviceContext* context) {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  PADDLE_ENFORCE(platform::dynload::cublasDgemm(
      reinterpret_cast<platform::CUDADeviceContext*>(context)->
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


}  // namespace math
}  // namespace operators
}  // namespace paddle
