/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "MathFunctions.h"
#include "hl_matrix_ops.cuh"
#include "hl_matrix_apply.cuh"

namespace paddle {

template<>
void gemm<float>(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
                 const int M, const int N, const int K,
                 const float alpha, const float* A, const int lda,
                 const float* B, const int ldb,
                 const float beta, float* C, const int ldc) {
  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template<>
void gemm<double>(const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB,
                  const int M, const int N, const int K,
                  const double alpha, const double* A, const int lda,
                  const double* B, const int ldb,
                  const double beta, double* C, const int ldc) {
  cblas_dgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template<>
int getrf<float>(const CBLAS_ORDER order, const int M, const int N,
                  float *A, const int lda, int *ipiv) {
#ifdef PADDLE_USE_ATLAS
  return clapack_sgetrf(order, M, N, A, lda, ipiv);
#else
  return LAPACKE_sgetrf(order, M, N, A, lda, ipiv);
#endif
}

template<>
int getrf<double>(const CBLAS_ORDER order, const int M, const int N,
                   double *A, const int lda, int *ipiv) {
#ifdef PADDLE_USE_ATLAS
  return clapack_dgetrf(order, M, N, A, lda, ipiv);
#else
  return LAPACKE_dgetrf(order, M, N, A, lda, ipiv);
#endif
}

template<>
int getri<float>(const CBLAS_ORDER order, const int N, float *A,
                  const int lda, const int *ipiv) {
#ifdef PADDLE_USE_ATLAS
  return clapack_sgetri(order, N, A, lda, ipiv);
#else
  return LAPACKE_sgetri(order, N, A, lda, ipiv);
#endif
}

template<>
int getri<double>(const CBLAS_ORDER order, const int N, double *A,
                  const int lda, const int *ipiv) {
#ifdef PADDLE_USE_ATLAS
  return clapack_dgetri(order, N, A, lda, ipiv);
#else
  return LAPACKE_dgetri(order, N, A, lda, ipiv);
#endif
}

template<>
void axpy<float>(const int n, const float alpha, const float* x, float* y) {
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

template<>
void axpy<double>(const int n, const double alpha, const double* x, double* y) {
  cblas_daxpy(n, alpha, x, 1, y, 1);
}

template<>
float dotProduct<float>(const int n, const float* x, const float* y) {
  return cblas_sdot(n, x, 1, y, 1);
}

template<>
double dotProduct<double>(const int n, const double* x, const double* y) {
  return cblas_ddot(n, x, 1, y, 1);
}

#ifdef PADDLE_USE_MKL

template<>
void vExp<float>(const int n, const float* a, float* r) {
  vsExp(n, a, r);
}

template<>
void vExp<double>(const int n, const double* a, double* r) {
  vdExp(n, a, r);
}

template<>
void vPow<float>(const int n, const float* a, const float b, float* r) {
  vsPowx(n, a, b, r);
}

template<>
void vPow<double>(const int n, const double* a, const double b, double* r) {
  vdPowx(n, a, b, r);
}

template<>
void vLog<float>(const int n, const float* a, float* r) {
  vsLn(n, a, r);
}

template<>
void vLog<double>(const int n, const double* a, double* r) {
  vdLn(n, a, r);
}

template<>
void vAdd<float>(const int n, const float* a, const float* b, float* r) {
  vsAdd(n, a, b, r);
}

template<>
void vAdd<double>(const int n, const double* a, const double* b, double* r) {
  vdAdd(n, a, b, r);
}

template<>
void vInvSqrt<float>(const int n, const float* a, float* r) {
  vsInvSqrt(n, a, r);
}

template<>
void vInvSqrt<double>(const int n, const double* a, double* r) {
  vdInvSqrt(n, a, r);
}

template<>
void vLog1p<float>(const int n, const float* a, float* r) {
  vsLog1p(n, a, r);
}

template<>
void vLog1p<double>(const int n, const double* a, double* r) {
  vdLog1p(n, a, r);
}

template<>
void vTanh<float>(const int n, const float* a, float* r) {
  vsTanh(n, a, r);
}

template<>
void vTanh<double>(const int n, const double* a, double* r) {
  vdTanh(n, a, r);
}
#else

DEFINE_MATRIX_BINARY_OP(vExp, b = std::exp(a));
template<class T>
void vExp(const int n, const T* a, T* r) {
  hl_cpu_apply_binary_op<T, binary::vExp<T>, 0, 0>(
    binary::vExp<T>(), const_cast<T*>(a), r, 1, n, n, n);
}

DEFINE_MATRIX_BINARY_OP(vLog, b = std::log(a));
template<class T>
void vLog(const int n, const T* a, T* r) {
  hl_cpu_apply_binary_op<T, binary::vLog<T>, 0, 0>(
    binary::vLog<T>(), const_cast<T*>(a), r, 1, n, n, n);
}

DEFINE_MATRIX_BINARY_OP(vInvSqrt, b = 1.0f / std::sqrt(a));
template<class T>
void vInvSqrt(const int n, const T* a, T* r) {
  hl_cpu_apply_binary_op<T, binary::vInvSqrt<T>, 0, 0>(
    binary::vInvSqrt<T>(), const_cast<T*>(a), r, 1, n, n, n);
}

DEFINE_MATRIX_BINARY_OP(vLog1p, b = std::log(1.0f + a));
template<class T>
void vLog1p(const int n, const T* a, T* r) {
  hl_cpu_apply_binary_op<T, binary::vLog1p<T>, 0, 0>(
    binary::vLog1p<T>(), const_cast<T*>(a), r, 1, n, n, n);
}

DEFINE_MATRIX_BINARY_OP(vTanh,
    T tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    b = 2.0 / (1.0 + std::exp(tmp)) - 1.0);
template<class T>
void vTanh(const int n, const T* a, T* r) {
  hl_cpu_apply_binary_op<T, binary::vTanh<T>, 0, 0>(
    binary::vTanh<T>(), const_cast<T*>(a), r, 1, n, n, n);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(vPow, ONE_PARAMETER, b = std::pow(a, p));
template<class T>
void vPow(const int n, const T* a, const T b, T* r) {
  hl_cpu_apply_binary_op<T, binary::vPow<T>, 0, 0>(
    binary::vPow<T>(b), const_cast<T*>(a), r, 1, n, n, n);
}

DEFINE_MATRIX_TERNARY_OP(vAdd, c = a + b);
template<class T>
void vAdd(const int n, const T* a, const T* b, T* r) {
  hl_cpu_apply_ternary_op<T, ternary::vAdd<T>, 0, 0>(ternary::vAdd<T>(),
    const_cast<T*>(a), const_cast<T*>(b), r, 1, n, n, n , n);
}

template void vExp(const int n, const float* a, float* r);
template void vExp(const int n, const double* a, double* r);
template void vLog(const int n, const float* a, float* r);
template void vLog(const int n, const double* a, double* r);
template void vInvSqrt(const int n, const double* a, double* r);
template void vInvSqrt(const int n, const float* a, float* r);
template void vLog1p(const int n, const float* a, float* r);
template void vLog1p(const int n, const double* a, double* r);
template void vTanh(const int n, const float* a, float* r);
template void vTanh(const int n, const double* a, double* r);
template void vPow(const int n, const float* a, const float b, float* r);
template void vPow(const int n, const double* a, const double b, double* r);
template void vAdd(const int n, const float* a, const float* b, float* r);
template void vAdd(const int n, const double* a, const double* b, double* r);

#endif

}  // namespace paddle
