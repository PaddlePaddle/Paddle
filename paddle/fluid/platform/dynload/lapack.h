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

#pragma once

#include <mutex>
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

// Note(zhouwei): because lapack doesn't provide appropriate header file
// getrf
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv,
                        int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv,
                        int *info);

// getrs
extern "C" void dgetrs_(char *trans, int *n, int *nrhs, double *a, int *lda,
                        int *ipiv, double *b, int *ldb, int *info);
extern "C" void sgetrs_(char *trans, int *n, int *nrhs, float *a, int *lda,
                        int *ipiv, float *b, int *ldb, int *info);

// potrf
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);

// geev
extern "C" void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda,
                       double *wr, double *wi, double *vl, int *ldvl,
                       double *vr, int *ldvr, double *work, int *lwork,
                       int *info);
extern "C" void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda,
                       float *wr, float *wi, float *vl, int *ldvl, float *vr,
                       int *ldvr, float *work, int *lwork, int *info);

// syevd
extern "C" void dsyevd_(char *jobz, char *uplo, int *n, double *a, int *lda,
                        double *w, double *work, int *lwork, int *iwork,
                        int *liwork, int *info);
extern "C" void ssyevd_(char *jobz, char *uplo, int *n, float *a, int *lda,
                        float *w, float *work, int *lwork, int *iwork,
                        int *liwork, int *info);

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag lapack_dso_flag;
extern void *lapack_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load lapack routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_LAPACK_WRAP(__name)                                     \
  struct DynLoad__##__name {                                                 \
    template <typename... Args>                                              \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {         \
      using lapackFunc = decltype(&::__name);                                \
      std::call_once(lapack_dso_flag, []() {                                 \
        lapack_dso_handle = paddle::platform::dynload::GetLAPACKDsoHandle(); \
      });                                                                    \
      static void *p_##_name = dlsym(lapack_dso_handle, #__name);            \
      return reinterpret_cast<lapackFunc>(p_##_name)(args...);               \
    }                                                                        \
  };                                                                         \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_LAPACK_WRAP(__name) \
  DYNAMIC_LOAD_LAPACK_WRAP(__name)

#define LAPACK_ROUTINE_EACH(__macro) \
  __macro(dgetrf_);                  \
  __macro(sgetrf_);                  \
  __macro(dgetrs_);                  \
  __macro(sgetrs_);                  \
  __macro(dpotrf_);                  \
  __macro(spotrf_);                  \
  __macro(dgeev_);                   \
  __macro(sgeev_);                   \
  __macro(dsyevd_);                  \
  __macro(ssyevd_);

LAPACK_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_LAPACK_WRAP);

#undef DYNAMIC_LOAD_LAPACK_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
