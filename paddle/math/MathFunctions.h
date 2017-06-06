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

#ifndef MATHFUNCTIONS_H_
#define MATHFUNCTIONS_H_

#ifdef PADDLE_USE_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#endif

#ifdef PADDLE_USE_ATLAS
extern "C" {
#include <cblas.h>
#include <clapack.h>
}
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#endif

#ifndef LAPACK_FOUND
extern "C" {
#include <cblas.h>
int LAPACKE_sgetrf(
    int matrix_layout, int m, int n, float* a, int lda, int* ipiv);
int LAPACKE_dgetrf(
    int matrix_layout, int m, int n, double* a, int lda, int* ipiv);
int LAPACKE_sgetri(
    int matrix_layout, int n, float* a, int lda, const int* ipiv);
int LAPACKE_dgetri(
    int matrix_layout, int n, double* a, int lda, const int* ipiv);
}
#endif

#include <cmath>

namespace paddle {

template <class T>
void gemm(const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB,
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

template <class T>
int getrf(const CBLAS_ORDER Order,
          const int M,
          const int N,
          T* A,
          const int lda,
          int* ipiv);

template <class T>
int getri(
    const CBLAS_ORDER Order, const int N, T* A, const int lda, const int* ipiv);

template <class T>
void axpy(const int n, const T alpha, const T* x, T* y);

template <class T>
T dotProduct(const int n, const T* x, const T* y);

template <class T>
void vExp(const int n, const T* a, T* r);

template <class T>
void vPow(const int n, const T* a, const T b, T* r);

template <class T>
void vLog(const int n, const T* a, T* r);

template <class T>
void vAdd(const int n, const T* a, const T* b, T* r);

template <class T>
void vInvSqrt(const int n, const T* a, T* r);

template <class T>
void vLog1p(const int n, const T* a, T* r);

template <class T>
void vTanh(const int n, const T* a, T* r);

}  // namespace paddle

#endif  // MATHFUNCTIONS_H_
