/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

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

#include <cublasLt.h>
#include <cuda.h>

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag cublasLt_dso_flag;
extern void *cublasLt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublasLt routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUBLASLT_WRAP(__name)                          \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using cublasLt_func =                                                 \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);         \
      std::call_once(cublasLt_dso_flag, []() {                              \
        cublasLt_dso_handle = phi::dynload::GetCublasLtDsoHandle();         \
      });                                                                   \
      static void *p_##__name = dlsym(cublasLt_dso_handle, #__name);        \
      return reinterpret_cast<cublasLt_func>(p_##__name)(args...);          \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

// APIs available after CUDA 11.1
#if CUDA_VERSION >= 11010
#define CUBLASLT_BLAS_ROUTINE_EACH(__macro)         \
  __macro(cublasLtCreate);                          \
  __macro(cublasLtDestroy);                         \
  __macro(cublasLtMatmul);                          \
  __macro(cublasLtMatmulDescCreate);                \
  __macro(cublasLtMatmulDescDestroy);               \
  __macro(cublasLtMatmulDescSetAttribute);          \
  __macro(cublasLtMatmulDescGetAttribute);          \
  __macro(cublasLtMatrixLayoutCreate);              \
  __macro(cublasLtMatrixLayoutDestroy);             \
  __macro(cublasLtMatrixLayoutSetAttribute);        \
  __macro(cublasLtMatrixLayoutGetAttribute);        \
  __macro(cublasLtMatmulPreferenceCreate);          \
  __macro(cublasLtMatmulPreferenceDestroy);         \
  __macro(cublasLtMatmulPreferenceSetAttribute);    \
  __macro(cublasLtMatmulAlgoGetHeuristic);          \
  __macro(cublasLtMatrixTransform);                 \
  __macro(cublasLtMatrixTransformDescCreate);       \
  __macro(cublasLtMatrixTransformDescDestroy);      \
  __macro(cublasLtMatrixTransformDescSetAttribute); \
  __macro(cublasLtMatmulAlgoInit);                  \
  __macro(cublasLtMatmulAlgoConfigSetAttribute);    \
  __macro(cublasLtMatmulAlgoGetIds);                \
  __macro(cublasLtMatmulAlgoCapGetAttribute);       \
  __macro(cublasLtMatmulAlgoCheck);                 \
  __macro(cublasLtGetCudartVersion);
#else
#define CUBLASLT_BLAS_ROUTINE_EACH(__macro)      \
  __macro(cublasLtCreate);                       \
  __macro(cublasLtDestroy);                      \
  __macro(cublasLtMatmul);                       \
  __macro(cublasLtMatmulDescCreate);             \
  __macro(cublasLtMatmulDescDestroy);            \
  __macro(cublasLtMatmulDescSetAttribute);       \
  __macro(cublasLtMatmulDescGetAttribute);       \
  __macro(cublasLtMatrixLayoutCreate);           \
  __macro(cublasLtMatrixLayoutDestroy);          \
  __macro(cublasLtMatrixLayoutSetAttribute);     \
  __macro(cublasLtMatrixLayoutGetAttribute);     \
  __macro(cublasLtMatmulPreferenceCreate);       \
  __macro(cublasLtMatmulPreferenceDestroy);      \
  __macro(cublasLtMatmulPreferenceSetAttribute); \
  __macro(cublasLtMatmulAlgoGetHeuristic);       \
  __macro(cublasLtMatrixTransform);              \
  __macro(cublasLtMatrixTransformDescCreate);    \
  __macro(cublasLtMatrixTransformDescDestroy);   \
  __macro(cublasLtMatrixTransformDescSetAttribute);
#endif

CUBLASLT_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUBLASLT_WRAP)
// #endif

#undef DECLARE_DYNAMIC_LOAD_CUBLASLT_WRAP
}  // namespace dynload
}  // namespace phi
