//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/fluid/platform/dynload/lapack.h"
#include "paddle/phi/common/complex.h"

namespace phi {
namespace funcs {

// LU (for example)
template <>
void lapackLu<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  paddle::platform::dynload::dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template <>
void lapackLu<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  paddle::platform::dynload::sgetrf_(&m, &n, a, &lda, ipiv, info);
}

// eigh
template <>
void lapackEigh<float>(char jobz,
                       char uplo,
                       int n,
                       float *a,
                       int lda,
                       float *w,
                       float *work,
                       int lwork,
                       float *rwork,
                       int lrwork,
                       int *iwork,
                       int liwork,
                       int *info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  paddle::platform::dynload::ssyevd_(
      &jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template <>
void lapackEigh<double>(char jobz,
                        char uplo,
                        int n,
                        double *a,
                        int lda,
                        double *w,
                        double *work,
                        int lwork,
                        double *rwork,
                        int lrwork,
                        int *iwork,
                        int liwork,
                        int *info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  paddle::platform::dynload::dsyevd_(
      &jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template <>
void lapackEigh<phi::dtype::complex<float>, float>(
    char jobz,
    char uplo,
    int n,
    phi::dtype::complex<float> *a,
    int lda,
    float *w,
    phi::dtype::complex<float> *work,
    int lwork,
    float *rwork,
    int lrwork,
    int *iwork,
    int liwork,
    int *info) {
  paddle::platform::dynload::cheevd_(
      &jobz,
      &uplo,
      &n,
      reinterpret_cast<std::complex<float> *>(a),
      &lda,
      w,
      reinterpret_cast<std::complex<float> *>(work),
      &lwork,
      rwork,
      &lrwork,
      iwork,
      &liwork,
      info);
}

template <>
void lapackEigh<phi::dtype::complex<double>, double>(
    char jobz,
    char uplo,
    int n,
    phi::dtype::complex<double> *a,
    int lda,
    double *w,
    phi::dtype::complex<double> *work,
    int lwork,
    double *rwork,
    int lrwork,
    int *iwork,
    int liwork,
    int *info) {
  paddle::platform::dynload::zheevd_(
      &jobz,
      &uplo,
      &n,
      reinterpret_cast<std::complex<double> *>(a),
      &lda,
      w,
      reinterpret_cast<std::complex<double> *>(work),
      &lwork,
      rwork,
      &lrwork,
      iwork,
      &liwork,
      info);
}

// Eig
template <>
void lapackEig<double>(char jobvl,
                       char jobvr,
                       int n,
                       double *a,
                       int lda,
                       double *w,
                       double *vl,
                       int ldvl,
                       double *vr,
                       int ldvr,
                       double *work,
                       int lwork,
                       double *rwork,
                       int *info) {
  double *wr = w;
  double *wi = w + n;
  (void)rwork;  // unused
  paddle::platform::dynload::dgeev_(&jobvl,
                                    &jobvr,
                                    &n,
                                    a,
                                    &lda,
                                    wr,
                                    wi,
                                    vl,
                                    &ldvl,
                                    vr,
                                    &ldvr,
                                    work,
                                    &lwork,
                                    info);
}

template <>
void lapackEig<float>(char jobvl,
                      char jobvr,
                      int n,
                      float *a,
                      int lda,
                      float *w,
                      float *vl,
                      int ldvl,
                      float *vr,
                      int ldvr,
                      float *work,
                      int lwork,
                      float *rwork,
                      int *info) {
  float *wr = w;
  float *wi = w + n;
  (void)rwork;  // unused
  paddle::platform::dynload::sgeev_(&jobvl,
                                    &jobvr,
                                    &n,
                                    a,
                                    &lda,
                                    wr,
                                    wi,
                                    vl,
                                    &ldvl,
                                    vr,
                                    &ldvr,
                                    work,
                                    &lwork,
                                    info);
}

template <>
void lapackEig<phi::dtype::complex<double>, double>(
    char jobvl,
    char jobvr,
    int n,
    phi::dtype::complex<double> *a,
    int lda,
    phi::dtype::complex<double> *w,
    phi::dtype::complex<double> *vl,
    int ldvl,
    phi::dtype::complex<double> *vr,
    int ldvr,
    phi::dtype::complex<double> *work,
    int lwork,
    double *rwork,
    int *info) {
  paddle::platform::dynload::zgeev_(
      &jobvl,
      &jobvr,
      &n,
      reinterpret_cast<std::complex<double> *>(a),
      &lda,
      reinterpret_cast<std::complex<double> *>(w),
      reinterpret_cast<std::complex<double> *>(vl),
      &ldvl,
      reinterpret_cast<std::complex<double> *>(vr),
      &ldvr,
      reinterpret_cast<std::complex<double> *>(work),
      &lwork,
      rwork,
      info);
}

template <>
void lapackEig<phi::dtype::complex<float>, float>(
    char jobvl,
    char jobvr,
    int n,
    phi::dtype::complex<float> *a,
    int lda,
    phi::dtype::complex<float> *w,
    phi::dtype::complex<float> *vl,
    int ldvl,
    phi::dtype::complex<float> *vr,
    int ldvr,
    phi::dtype::complex<float> *work,
    int lwork,
    float *rwork,
    int *info) {
  paddle::platform::dynload::cgeev_(
      &jobvl,
      &jobvr,
      &n,
      reinterpret_cast<std::complex<float> *>(a),
      &lda,
      reinterpret_cast<std::complex<float> *>(w),
      reinterpret_cast<std::complex<float> *>(vl),
      &ldvl,
      reinterpret_cast<std::complex<float> *>(vr),
      &ldvr,
      reinterpret_cast<std::complex<float> *>(work),
      &lwork,
      rwork,
      info);
}

template <>
void lapackGels<double>(char trans,
                        int m,
                        int n,
                        int nrhs,
                        double *a,
                        int lda,
                        double *b,
                        int ldb,
                        double *work,
                        int lwork,
                        int *info) {
  paddle::platform::dynload::dgels_(
      &trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
}

template <>
void lapackGels<float>(char trans,
                       int m,
                       int n,
                       int nrhs,
                       float *a,
                       int lda,
                       float *b,
                       int ldb,
                       float *work,
                       int lwork,
                       int *info) {
  paddle::platform::dynload::sgels_(
      &trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
}

template <>
void lapackGelsd<double>(int m,
                         int n,
                         int nrhs,
                         double *a,
                         int lda,
                         double *b,
                         int ldb,
                         double *s,
                         double rcond,
                         int *rank,
                         double *work,
                         int lwork,
                         double *rwork,
                         int *iwork,
                         int *info) {
  paddle::platform::dynload::dgelsd_(&m,
                                     &n,
                                     &nrhs,
                                     a,
                                     &lda,
                                     b,
                                     &ldb,
                                     s,
                                     &rcond,
                                     rank,
                                     work,
                                     &lwork,
                                     iwork,
                                     info);
}

template <>
void lapackGelsd<float>(int m,
                        int n,
                        int nrhs,
                        float *a,
                        int lda,
                        float *b,
                        int ldb,
                        float *s,
                        float rcond,
                        int *rank,
                        float *work,
                        int lwork,
                        float *rwork,
                        int *iwork,
                        int *info) {
  paddle::platform::dynload::sgelsd_(&m,
                                     &n,
                                     &nrhs,
                                     a,
                                     &lda,
                                     b,
                                     &ldb,
                                     s,
                                     &rcond,
                                     rank,
                                     work,
                                     &lwork,
                                     iwork,
                                     info);
}

template <>
void lapackGelsy<double>(int m,
                         int n,
                         int nrhs,
                         double *a,
                         int lda,
                         double *b,
                         int ldb,
                         int *jpvt,
                         double rcond,
                         int *rank,
                         double *work,
                         int lwork,
                         double *rwork,
                         int *info) {
  paddle::platform::dynload::dgelsy_(
      &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work, &lwork, info);
}

template <>
void lapackGelsy<float>(int m,
                        int n,
                        int nrhs,
                        float *a,
                        int lda,
                        float *b,
                        int ldb,
                        int *jpvt,
                        float rcond,
                        int *rank,
                        float *work,
                        int lwork,
                        float *rwork,
                        int *info) {
  paddle::platform::dynload::sgelsy_(
      &m, &n, &nrhs, a, &lda, b, &ldb, jpvt, &rcond, rank, work, &lwork, info);
}

template <>
void lapackGelss<double>(int m,
                         int n,
                         int nrhs,
                         double *a,
                         int lda,
                         double *b,
                         int ldb,
                         double *s,
                         double rcond,
                         int *rank,
                         double *work,
                         int lwork,
                         double *rwork,
                         int *info) {
  paddle::platform::dynload::dgelss_(
      &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork, info);
}

template <>
void lapackGelss<float>(int m,
                        int n,
                        int nrhs,
                        float *a,
                        int lda,
                        float *b,
                        int ldb,
                        float *s,
                        float rcond,
                        int *rank,
                        float *work,
                        int lwork,
                        float *rwork,
                        int *info) {
  paddle::platform::dynload::sgelss_(
      &m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, rank, work, &lwork, info);
}

template <>
void lapackCholeskySolve<phi::dtype::complex<double>>(
    char uplo,
    int n,
    int nrhs,
    phi::dtype::complex<double> *a,
    int lda,
    phi::dtype::complex<double> *b,
    int ldb,
    int *info) {
  paddle::platform::dynload::zpotrs_(
      &uplo,
      &n,
      &nrhs,
      reinterpret_cast<std::complex<double> *>(a),
      &lda,
      reinterpret_cast<std::complex<double> *>(b),
      &ldb,
      info);
}

template <>
void lapackCholeskySolve<phi::dtype::complex<float>>(
    char uplo,
    int n,
    int nrhs,
    phi::dtype::complex<float> *a,
    int lda,
    phi::dtype::complex<float> *b,
    int ldb,
    int *info) {
  paddle::platform::dynload::cpotrs_(&uplo,
                                     &n,
                                     &nrhs,
                                     reinterpret_cast<std::complex<float> *>(a),
                                     &lda,
                                     reinterpret_cast<std::complex<float> *>(b),
                                     &ldb,
                                     info);
}

template <>
void lapackCholeskySolve<double>(char uplo,
                                 int n,
                                 int nrhs,
                                 double *a,
                                 int lda,
                                 double *b,
                                 int ldb,
                                 int *info) {
  paddle::platform::dynload::dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template <>
void lapackCholeskySolve<float>(char uplo,
                                int n,
                                int nrhs,
                                float *a,
                                int lda,
                                float *b,
                                int ldb,
                                int *info) {
  paddle::platform::dynload::spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

}  // namespace funcs
}  // namespace phi
