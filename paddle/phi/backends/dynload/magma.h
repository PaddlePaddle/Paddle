/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_MAGMA
#pragma once

#ifdef PADDLE_WITH_HIP
#include <hip/hip_complex.h>
#include <thrust/complex.h>
typedef hipDoubleComplex magmaDoubleComplex;
typedef hipFloatComplex magmaFloatComplex;
#endif  // PADDLE_WITH_HIP

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_XPU)
#include <cuComplex.h>
#include <complex>
typedef cuDoubleComplex magmaDoubleComplex;
typedef cuFloatComplex magmaFloatComplex;
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_XPU

#include <mutex>
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

typedef int magma_int_t;

typedef enum { MagmaNoVec = 301, MagmaVec = 302 } magma_vec_t;

// geev
extern "C" magma_int_t magma_dgeev(magma_vec_t jobvl,
                                   magma_vec_t jobvr,
                                   magma_int_t n,
                                   double *A,
                                   magma_int_t lda,
                                   double *wr,
                                   double *wi,
                                   double *VL,
                                   magma_int_t ldvl,
                                   double *VR,
                                   magma_int_t ldvr,
                                   double *work,
                                   magma_int_t lwork,
                                   magma_int_t *info);

// real float
extern "C" magma_int_t magma_sgeev(magma_vec_t jobvl,
                                   magma_vec_t jobvr,
                                   magma_int_t n,
                                   float *A,
                                   magma_int_t lda,
                                   float *wr,
                                   float *wi,
                                   float *VL,
                                   magma_int_t ldvl,
                                   float *VR,
                                   magma_int_t ldvr,
                                   float *work,
                                   magma_int_t lwork,
                                   magma_int_t *info);

// complex double
extern "C" magma_int_t magma_zgeev(magma_vec_t jobvl,
                                   magma_vec_t jobvr,
                                   magma_int_t n,
                                   magmaDoubleComplex *A,
                                   magma_int_t lda,
                                   magmaDoubleComplex *w,
                                   magmaDoubleComplex *VL,
                                   magma_int_t ldvl,
                                   magmaDoubleComplex *VR,
                                   magma_int_t ldvr,
                                   magmaDoubleComplex *work,
                                   magma_int_t lwork,
                                   double *rwork,
                                   magma_int_t *info);

// complex float
extern "C" magma_int_t magma_cgeev(magma_vec_t jobvl,
                                   magma_vec_t jobvr,
                                   magma_int_t n,
                                   magmaFloatComplex *A,
                                   magma_int_t lda,
                                   magmaFloatComplex *w,
                                   magmaFloatComplex *VL,
                                   magma_int_t ldvl,
                                   magmaFloatComplex *VR,
                                   magma_int_t ldvr,
                                   magmaFloatComplex *work,
                                   magma_int_t lwork,
                                   float *rwork,
                                   magma_int_t *info);

extern "C" magma_int_t magma_init();
extern "C" magma_int_t magma_finalize();

namespace phi {
namespace dynload {

extern std::once_flag magma_dso_flag;
extern void *magma_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load magma routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_MAGMA_WRAP(__name)                              \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using magmaFunc = decltype(&::__name);                         \
      std::call_once(magma_dso_flag, []() {                          \
        magma_dso_handle = phi::dynload::GetMAGMADsoHandle();        \
      });                                                            \
      static void *p_##_name = dlsym(magma_dso_handle, #__name);     \
      return reinterpret_cast<magmaFunc>(p_##_name)(args...);        \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_MAGMA_WRAP(__name) DYNAMIC_LOAD_MAGMA_WRAP(__name)

#define MAGMA_ROUTINE_EACH(__macro) \
  __macro(magma_dgeev);             \
  __macro(magma_sgeev);             \
  __macro(magma_zgeev);             \
  __macro(magma_cgeev);             \
  __macro(magma_init);              \
  __macro(magma_finalize);

MAGMA_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MAGMA_WRAP);

#undef DYNAMIC_LOAD_MAGMA_WRAP

}  // namespace dynload
}  // namespace phi

#endif
