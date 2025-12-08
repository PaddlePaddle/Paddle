//   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_MAGMA
#include "paddle/phi/kernels/funcs/magma/magma_function.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/backends/dynload/magma.h"
#include "paddle/phi/common/data_type.h"

namespace phi::funcs {

void magmaEnsureInit() {
  static std::once_flag magma_once_flag;

  std::call_once(magma_once_flag, []() {
    magma_int_t info = dynload::magma_init();
    PADDLE_ENFORCE_EQ(
        info,
        0,
        phi::errors::External("magma_init failed, info code = %d,"
                              "please checkout this code in: "
                              "https://github.com/icl-utk-edu/magma/blob/"
                              "master/include/magma_types.h#L542",
                              info));

    std::atexit([]() {
      magma_int_t info = dynload::magma_finalize();
      PADDLE_ENFORCE_EQ(
          info,
          0,
          phi::errors::External("magma_finalize failed, info code = %d,"
                                "please checkout this code in: "
                                "https://github.com/icl-utk-edu/magma/blob/"
                                "master/include/magma_types.h#L542",
                                info));
    });
  });
}

// Eig
template <>
void magmaEig<double>(magma_vec_t jobvl,
                      magma_vec_t jobvr,
                      magma_int_t n,
                      double *a,
                      magma_int_t lda,
                      double *w,
                      double *vl,
                      magma_int_t ldvl,
                      double *vr,
                      magma_int_t ldvr,
                      double *work,
                      magma_int_t lwork,
                      double *rwork,
                      magma_int_t *info) {
  double *wr = w;
  double *wi = w + n;
  (void)rwork;  // unused
  *info = dynload::magma_dgeev(
      jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

template <>
void magmaEig<float>(magma_vec_t jobvl,
                     magma_vec_t jobvr,
                     magma_int_t n,
                     float *a,
                     magma_int_t lda,
                     float *w,
                     float *vl,
                     magma_int_t ldvl,
                     float *vr,
                     magma_int_t ldvr,
                     float *work,
                     magma_int_t lwork,
                     float *rwork,
                     magma_int_t *info) {
  float *wr = w;
  float *wi = w + n;
  (void)rwork;  // unused
  *info = dynload::magma_sgeev(
      jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

template <>
void magmaEig<phi::complex128, double>(magma_vec_t jobvl,
                                       magma_vec_t jobvr,
                                       magma_int_t n,
                                       phi::complex128 *a,
                                       magma_int_t lda,
                                       phi::complex128 *w,
                                       phi::complex128 *vl,
                                       magma_int_t ldvl,
                                       phi::complex128 *vr,
                                       magma_int_t ldvr,
                                       phi::complex128 *work,
                                       magma_int_t lwork,
                                       double *rwork,
                                       magma_int_t *info) {
  *info = dynload::magma_zgeev(jobvl,
                               jobvr,
                               n,
                               reinterpret_cast<magmaDoubleComplex *>(a),
                               lda,
                               reinterpret_cast<magmaDoubleComplex *>(w),
                               reinterpret_cast<magmaDoubleComplex *>(vl),
                               ldvl,
                               reinterpret_cast<magmaDoubleComplex *>(vr),
                               ldvr,
                               reinterpret_cast<magmaDoubleComplex *>(work),
                               lwork,
                               rwork,
                               info);
}

template <>
void magmaEig<phi::complex64, float>(magma_vec_t jobvl,
                                     magma_vec_t jobvr,
                                     magma_int_t n,
                                     phi::complex64 *a,
                                     magma_int_t lda,
                                     phi::complex64 *w,
                                     phi::complex64 *vl,
                                     magma_int_t ldvl,
                                     phi::complex64 *vr,
                                     magma_int_t ldvr,
                                     phi::complex64 *work,
                                     magma_int_t lwork,
                                     float *rwork,
                                     magma_int_t *info) {
  *info = dynload::magma_cgeev(jobvl,
                               jobvr,
                               n,
                               reinterpret_cast<magmaFloatComplex *>(a),
                               lda,
                               reinterpret_cast<magmaFloatComplex *>(w),
                               reinterpret_cast<magmaFloatComplex *>(vl),
                               ldvl,
                               reinterpret_cast<magmaFloatComplex *>(vr),
                               ldvr,
                               reinterpret_cast<magmaFloatComplex *>(work),
                               lwork,
                               rwork,
                               info);
}

}  // namespace phi::funcs
#endif
