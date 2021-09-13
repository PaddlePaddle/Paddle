//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/math/lapack_function.h"
#include "paddle/fluid/platform/dynload/lapack.h"

namespace paddle {
namespace operators {
namespace math {

// LU
template <>
void lapackLu<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  platform::dynload::dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template <>
void lapackLu<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  platform::dynload::sgetrf_(&m, &n, a, &lda, ipiv, info);
}

// Solve
template <>
void lapackLuSolve<double>(char trans, int n, int nrhs, double *a, int lda,
                           int *ipiv, double *b, int ldb, int *info) {
  platform::dynload::dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void lapackLuSolve<float>(char trans, int n, int nrhs, float *a, int lda,
                          int *ipiv, float *b, int ldb, int *info) {
  platform::dynload::sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

// Cholesky
template <>
void lapackCholesky<double>(char uplo, int n, double *a, int lda, int *info) {
  platform::dynload::dpotrf_(&uplo, &n, a, &lda, info);
}

template <>
void lapackCholesky<float>(char uplo, int n, float *a, int lda, int *info) {
  platform::dynload::spotrf_(&uplo, &n, a, &lda, info);
}

// Eig
template <>
void lapackEig<double>(char jobvl, char jobvr, int n, double *a, int lda,
                       double *w, double *vl, int ldvl, double *vr, int ldvr,
                       double *work, int lwork, double *rwork, int *info) {
  double *wr = w;
  double *wi = w + n;
  (void)rwork;  // unused
  platform::dynload::dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr,
                            &ldvr, work, &lwork, info);
}

template <>
void lapackEig<float>(char jobvl, char jobvr, int n, float *a, int lda,
                      float *w, float *vl, int ldvl, float *vr, int ldvr,
                      float *work, int lwork, float *rwork, int *info) {
  float *wr = w;
  float *wi = w + n;
  (void)rwork;  // unused
  platform::dynload::sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr,
                            &ldvr, work, &lwork, info);
}

// Syevd
template <>
void lapackSyevd<double>(char jobz, char uplo, int n, double *a, int lda,
                         double *w, double *work, int lwork, double *rwork,
                         int lrwork, int *iwork, int liwork, int *info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  platform::dynload::dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                             &liwork, info);
}

template <>
void lapackSyevd<float>(char jobz, char uplo, int n, float *a, int lda,
                        float *w, float *work, int lwork, float *rwork,
                        int lrwork, int *iwork, int liwork, int *info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  platform::dynload::ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                             &liwork, info);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
