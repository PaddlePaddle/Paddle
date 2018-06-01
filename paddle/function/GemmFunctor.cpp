/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "GemmFunctor.h"
#include "paddle/math/MathFunctions.h"

namespace paddle {

template <class T>
struct BlasGemm<DEVICE_TYPE_CPU, T> {
  static void compute(const bool transA,
                      const bool transB,
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
#ifdef PADDLE_USE_EIGEN_FOR_BLAS
    EigenBlasGemm<T>::compute(
        transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    gemm<T>(transA == false ? CblasNoTrans : CblasTrans,
            transB == false ? CblasNoTrans : CblasTrans,
            M,
            N,
            K,
            alpha,
            A,
            lda,
            B,
            ldb,
            beta,
            C,
            ldc);
#endif
  }
};

template <class T>
struct BlasGemm<DEVICE_TYPE_GPU, T> {
  static void compute(const bool transA,
                      const bool transB,
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
                  transA == false ? HPPL_OP_N : HPPL_OP_T,
                  (T*)B,
                  transB == false ? HPPL_OP_N : HPPL_OP_T,
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

template struct BlasGemm<DEVICE_TYPE_CPU, real>;
template struct BlasGemm<DEVICE_TYPE_GPU, real>;

}  // namespace paddle
