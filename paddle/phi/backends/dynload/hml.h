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

#pragma once

#include <hml.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag hml_dso_flag;
extern void *hml_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load hml routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_HML_WRAP(__name)                                \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using hmlFunc = decltype(&::__name);                           \
      std::call_once(hml_dso_flag, []() {                            \
        hml_dso_handle = phi::dynload::GetHMLDsoHandle();            \
      });                                                            \
      static void *p_##_name = dlsym(hml_dso_handle, #__name);       \
      return reinterpret_cast<hmlFunc>(p_##_name)(args...);          \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_HML_WRAP(__name) DYNAMIC_LOAD_HML_WRAP(__name)

#define HML_ROUTINE_EACH(__macro)    \
  __macro(cblas_sgemm);              \
  __macro(cblas_dgemm);              \
  __macro(cblas_cgemm);              \
  __macro(cblas_zgemm);              \
  __macro(cblas_saxpy);              \
  __macro(cblas_daxpy);              \
  __macro(cblas_caxpy);              \
  __macro(cblas_zaxpy);              \
  __macro(cblas_scopy);              \
  __macro(cblas_dcopy);              \
  __macro(cblas_ccopy);              \
  __macro(cblas_zcopy);              \
  __macro(cblas_sgemv);              \
  __macro(cblas_dgemv);              \
  __macro(cblas_cgemv);              \
  __macro(cblas_zgemv);              \
  __macro(cblas_strsm);              \
  __macro(cblas_dtrsm);              \
  __macro(cblas_ctrsm);              \
  __macro(cblas_ztrsm);              \
  __macro(cblas_sgemm_batch);        \
  __macro(cblas_dgemm_batch);        \
  __macro(cblas_cgemm_batch);        \
  __macro(cblas_zgemm_batch);        \
  __macro(cblas_sdot);               \
  __macro(cblas_ddot);               \
  __macro(cblas_sasum);              \
  __macro(cblas_dasum);              \
  __macro(cblas_isamax);             \
  __macro(cblas_idamax);             \
  __macro(cblas_sscal);              \
  __macro(cblas_dscal);              \
  __macro(vsAdd);                    \
  __macro(vdAdd);                    \
  __macro(vsSub);                    \
  __macro(vdSub);                    \
  __macro(vsMul);                    \
  __macro(vdMul);                    \
  __macro(vsDiv);                    \
  __macro(vdDiv);                    \
  __macro(vsExp);                    \
  __macro(vdExp);                    \
  __macro(vsSqr);                    \
  __macro(vdSqr);                    \
  __macro(vsPowx);                   \
  __macro(vdPowx);                   \
  __macro(vsInv);                    \
  __macro(vdInv);                    \
  __macro(hml_blas_set_num_threads); \
  __macro(hml_blas_get_num_threads);

HML_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HML_WRAP);

#undef DYNAMIC_LOAD_HML_WRAP

}  // namespace dynload
}  // namespace phi
