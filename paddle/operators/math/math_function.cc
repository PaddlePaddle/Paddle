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
void gemm<platform::CPUPlace, float>(const CBLAS_TRANSPOSE transA,
                                     const CBLAS_TRANSPOSE transB, const int M,
                                     const int N, const int K,
                                     const float alpha, const float* A,
                                     const float* B, const float beta, float* C,
                                     platform::DeviceContext* context) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void gemm<platform::CPUPlace, double>(const CBLAS_TRANSPOSE transA,
                                      const CBLAS_TRANSPOSE transB, const int M,
                                      const int N, const int K,
                                      const double alpha, const double* A,
                                      const double* B, const double beta,
                                      double* C,
                                      platform::DeviceContext* context) {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void matmul<platform::CPUPlace, float>(const framework::Tensor& matrix_a,
                                       bool trans_a,
                                       const framework::Tensor& matrix_b,
                                       bool trans_b, float alpha,
                                       framework::Tensor* matrix_out,
                                       float beta,
                                       platform::DeviceContext* context) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_cpu_place(matrix_a.place()) &&
                     platform::is_cpu_place(matrix_b.place()) &&
                     platform::is_cpu_place(matrix_out->place()),
                 "Matrix must all be in CPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::CPUPlace, float>(
      transA, transB, M, N, K, alpha, matrix_a.data<float>(),
      matrix_b.data<float>(), beta, matrix_out->data<float>(), context);
}

template <>
void matmul<platform::CPUPlace, double>(const framework::Tensor& matrix_a,
                                        bool trans_a,
                                        const framework::Tensor& matrix_b,
                                        bool trans_b, double alpha,
                                        framework::Tensor* matrix_out,
                                        double beta,
                                        platform::DeviceContext* context) {
  auto dim_a = matrix_a.dims();
  auto dim_b = matrix_b.dims();
  auto dim_out = matrix_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_cpu_place(matrix_a.place()) &&
                     platform::is_cpu_place(matrix_b.place()) &&
                     platform::is_cpu_place(matrix_out->place()),
                 "Matrix must all be in CPUPlace");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = (trans_a == false) ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::CPUPlace, double>(
      transA, transB, M, N, K, alpha, matrix_a.data<double>(),
      matrix_b.data<double>(), beta, matrix_out->data<double>(), context);
}

template <>
void copy_matrix<platform::CPUPlace, float>(const float* a, size_t a_offset,
                                            float* b, size_t b_offset,
                                            size_t len, size_t before,
                                            size_t after) {
  for (size_t i = 0; i < before; i++) {
    const float* src = a + a_offset * after * i;
    float* dest = b + b_offset * after * i;
    memcpy(dest, src, len);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
