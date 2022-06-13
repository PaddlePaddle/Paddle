/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <cuda.h>
#include <cusparseLt.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag cusparselt_dso_flag;
extern void *cusparselt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cupti routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUSPARSELT_WRAP(__name)                    \
  struct DynLoad__##__name {                                            \
    template <typename... Args>                                         \
    cusparseStatus_t operator()(Args... args) {                         \
      using cusparseltFunc = decltype(&::__name);                       \
      std::call_once(cusparselt_dso_flag, []() {                        \
        cusparselt_dso_handle = phi::dynload::GetCusparseLtDsoHandle(); \
      });                                                               \
      static void *p_##__name = dlsym(cusparselt_dso_handle, #__name);  \
      return reinterpret_cast<cusparseltFunc>(p_##__name)(args...);     \
    }                                                                   \
  };                                                                    \
  extern DynLoad__##__name __name
#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 11020
#define CUSPARSELT_ROUTINE_EACH(__macro)       \
  __macro(cusparseLtInit);                     \
  __macro(cusparseLtDestroy);                  \
  __macro(cusparseLtDenseDescriptorInit);      \
  __macro(cusparseLtStructuredDescriptorInit); \
  __macro(cusparseLtMatmulDescriptorInit);     \
  __macro(cusparseLtMatmulDescSetAttribute);   \
  __macro(cusparseLtMatmulAlgSelectionInit);   \
  __macro(cusparseLtMatmulAlgSetAttribute);    \
  __macro(cusparseLtMatmulGetWorkspace);       \
  __macro(cusparseLtMatmulPlanInit);           \
  __macro(cusparseLtMatDescriptorDestroy);     \
  __macro(cusparseLtSpMMACompressedSize2);     \
  __macro(cusparseLtSpMMACompress2);           \
  __macro(cusparseLtMatmulSearch);             \
  __macro(cusparseLtMatmulAlgGetAttribute);    \
  __macro(cusparseLtMatmulPlanDestroy);        \
  __macro(cusparseLtMatmul);                   \
  __macro(cusparseGetErrorString);

CUSPARSELT_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUSPARSELT_WRAP);
#endif
#endif

#undef DECLARE_DYNAMIC_LOAD_CUSPARSELT_WRAP
}  // namespace dynload
}  // namespace phi
