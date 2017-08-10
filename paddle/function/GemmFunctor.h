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

#pragma once

#include "paddle/math/MathFunctions.h"

namespace paddle {

// TODO(hedaoyuan): Since the hl_matrix_mul interface does not conform to the
// cblas_dgemm interface's parameter format, it is necessary to introduce
// GemmFunctor as a new interface. Later, when considering the implementation
// of MatMulFunction, we need to consider the reconstruction of hl_matrix_mul
// interface.
template <DeviceType Device, class T>
class GemmFunctor {
public:
  void operator()(const CBLAS_TRANSPOSE transA,
                  const CBLAS_TRANSPOSE TransB,
                  const int M,
                  const int N,
                  const int K,
                  const T alpha,
                  const T* A,
                  const int lda,
                  const T* B,
                  const int ldb,
                  const T beta,
                  T* C,
                  const int ldc);
};

template <class T>
class GemmFunctor<DEVICE_TYPE_CPU, T> {
public:
  void operator()(const CBLAS_TRANSPOSE transA,
                  const CBLAS_TRANSPOSE TransB,
                  const int M,
                  const int N,
                  const int K,
                  const T alpha,
                  const T* A,
                  const int lda,
                  const T* B,
                  const int ldb,
                  const T beta,
                  T* C,
                  const int ldc) {
    gemm<T>(transA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

template <class T>
class GemmFunctor<DEVICE_TYPE_GPU, T> {
public:
  void operator()(const CBLAS_TRANSPOSE transA,
                  const CBLAS_TRANSPOSE TransB,
                  const int M,
                  const int N,
                  const int K,
                  const T alpha,
                  const T* A,
                  const int lda,
                  const T* B,
                  const int ldb,
                  const T beta,
                  T* C,
                  const int ldc) {
    hl_matrix_mul((T*)A,
                  transA == CblasNoTrans ? HPPL_OP_N : HPPL_OP_T,
                  (T*)B,
                  TransB == CblasNoTrans ? HPPL_OP_N : HPPL_OP_T,
                  C,
                  M,
                  N,
                  K,
                  alpha,
                  beta,
                  lda,
                  ldb,
                  ldc);
  }
};

}  // namespace paddle
