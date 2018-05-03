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

#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CBlas;

template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_sgemm(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_dgemm(args...);
  }
};

template <>
struct CBlas<platform::float16> {
  static void GEMM(...) { PADDLE_THROW("float16 GEMM not supported on CPU"); }
};

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                            CBLAS_TRANSPOSE transB, int M,
                                            int N, int K, T alpha, const T *A,
                                            const T *B, T beta, T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  CBlas<T>::GEMM(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM(bool transA, bool transB, int M,
                                            int N, int K, T alpha, const T *A,
                                            int lda, const T *B, int ldb,
                                            T beta, T *C, int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor, transA == false ? CblasNoTrans : CblasTrans,
                 transB == false ? CblasNoTrans : CblasTrans, M, N, K, alpha, A,
                 lda, B, ldb, beta, C, ldc);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const framework::Tensor &mat_a, bool trans_a,
                                 const framework::Tensor &mat_b, bool trans_b,
                                 T alpha, framework::Tensor *mat_out,
                                 T beta) const {
  auto dim_a = mat_a.dims();
  auto dim_b = mat_b.dims();
  auto dim_out = mat_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");
  PADDLE_ENFORCE(
      mat_a.place() == mat_b.place() && mat_a.place() == mat_out->place(),
      "The places of matrices must be same");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = !trans_a ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !trans_b ? CblasNoTrans : CblasTrans;

  this->GEMM(transA, transB, M, N, K, alpha, mat_a.data<T>(), mat_b.data<T>(),
             beta, mat_out->data<T>());
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
