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
  int lda = K;
  int ldb = N;
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
  int lda = K;
  int ldb = N;
  int ldc = N;
  cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template <>
void matmul<platform::CPUPlace, float>(const framework::Tensor& in1, bool in1_T,
                                       const framework::Tensor& in2, bool in2_T,
                                       float alpha, framework::Tensor* out,
                                       float beta,
                                       platform::DeviceContext* context) {
  auto in1_dim = in1.dims();
  auto in2_dim = in2.dims();
  auto out_dim = out->dims();
  PADDLE_ENFORCE(
      in1_dim.size() == 2 && in2_dim.size() == 2 && out_dim.size() == 2,
      "The input and output of matmul be matrix");

  PADDLE_ENFORCE(platform::is_cpu_place(in1.place()) &&
                     platform::is_cpu_place(in2.place()) &&
                     platform::is_cpu_place(out->place()),
                 "Matrix must all be in CPUPlace");

  int M = out_dim[0];
  int N = out_dim[1];
  int K = (in1_T == false) ? in1_dim[1] : in1_dim[0];

  CBLAS_TRANSPOSE in1_Trans = (in1_T == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE in2_Trans = (in2_T == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::CPUPlace, float>(in1_Trans, in2_Trans, M, N, K, alpha,
                                  in1.data<float>(), in2.data<float>(), beta,
                                  out->data<float>(), context);
}

template <>
void matmul<platform::CPUPlace, double>(const framework::Tensor& in1,
                                        bool in1_T,
                                        const framework::Tensor& in2,
                                        bool in2_T, float alpha,
                                        framework::Tensor* out, float beta,
                                        platform::DeviceContext* context) {
  auto in1_dim = in1.dims();
  auto in2_dim = in2.dims();
  auto out_dim = out->dims();
  PADDLE_ENFORCE(
      in1_dim.size() == 2 && in2_dim.size() == 2 && out_dim.size() == 2,
      "The input and output of matmul be matrix");
  PADDLE_ENFORCE(platform::is_cpu_place(in1.place()) &&
                     platform::is_cpu_place(in2.place()) &&
                     platform::is_cpu_place(out->place()),
                 "Matrix must all be in CPUPlace");

  int M = out_dim[0];
  int N = out_dim[1];
  int K = (in1_T == false) ? in1_dim[1] : in1_dim[0];
  CBLAS_TRANSPOSE in1_Trans = (in1_T == false) ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE in2_Trans = (in2_T == false) ? CblasNoTrans : CblasTrans;

  gemm<platform::CPUPlace, double>(in1_Trans, in2_Trans, M, N, K, alpha,
                                   in1.data<double>(), in2.data<double>(), beta,
                                   out->data<double>(), context);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
