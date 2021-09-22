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

// LU (for example)
template <>
void lapackLu<double>(int m, int n, double* a, int lda, int* ipiv, int* info) {
  platform::dynload::dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template <>
void lapackLu<float>(int m, int n, float* a, int lda, int* ipiv, int* info) {
  platform::dynload::sgetrf_(&m, &n, a, &lda, ipiv, info);
}

template <>
void lapackEvd<float, float>(char jobz, char uplo, int n, float* a, int lda,
                             float* w, float* work, int lwork, float* rwork,
                             int lrwork, int* iwork, int liwork, int* info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  platform::dynload::ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                             &liwork, info);
}

template <>
void lapackEvd<double, double>(char jobz, char uplo, int n, double* a, int lda,
                               double* w, double* work, int lwork,
                               double* rwork, int lrwork, int* iwork,
                               int liwork, int* info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  platform::dynload::dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork,
                             &liwork, info);
}

template <>
void lapackEvd<paddle::platform::complex<float>, float>(
    char jobz, char uplo, int n, paddle::platform::complex<float>* a, int lda,
    float* w, paddle::platform::complex<float>* work, int lwork, float* rwork,
    int lrwork, int* iwork, int liwork, int* info) {
  platform::dynload::cheevd_(&jobz, &uplo, &n,
                             reinterpret_cast<std::complex<float>*>(a), &lda, w,
                             reinterpret_cast<std::complex<float>*>(work),
                             &lwork, rwork, &lrwork, iwork, &liwork, info);
}

template <>
void lapackEvd<paddle::platform::complex<double>, double>(
    char jobz, char uplo, int n, paddle::platform::complex<double>* a, int lda,
    double* w, paddle::platform::complex<double>* work, int lwork,
    double* rwork, int lrwork, int* iwork, int liwork, int* info) {
  platform::dynload::zheevd_(&jobz, &uplo, &n,
                             reinterpret_cast<std::complex<double>*>(a), &lda,
                             w, reinterpret_cast<std::complex<double>*>(work),
                             &lwork, rwork, &lrwork, iwork, &liwork, info);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
