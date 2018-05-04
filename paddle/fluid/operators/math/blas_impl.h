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
void Blas<platform::CPUDeviceContext>::GEMM(const CBLAS_TRANSPOSE transA,
                                            const CBLAS_TRANSPOSE transB,
                                            const int M, const int N,
                                            const int K, const T alpha,
                                            const T *A, const T *B,
                                            const T beta, T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  CBlas<T>::GEMM(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM(
    const bool transA, const bool transB, const int M, const int N, const int K,
    const T alpha, const T *A, const int lda, const T *B, const int ldb,
    const T beta, T *C, const int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor, transA == false ? CblasNoTrans : CblasTrans,
                 transB == false ? CblasNoTrans : CblasTrans, M, N, K, alpha, A,
                 lda, B, ldb, beta, C, ldc);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
