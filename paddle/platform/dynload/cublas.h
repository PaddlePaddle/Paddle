/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <cublas_v2.h>
#include <dlfcn.h>
#include <mutex>
#include "paddle/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag cublas_dso_flag;
extern void *cublas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)                    \
  struct DynLoad__##__name {                                        \
    template <typename... Args>                                     \
    inline cublasStatus_t operator()(Args... args) {                \
      typedef cublasStatus_t (*cublasFunc)(Args...);                \
      std::call_once(cublas_dso_flag,                               \
                     paddle::platform::dynload::GetCublasDsoHandle, \
                     &cublas_dso_handle);                           \
      void *p_##__name = dlsym(cublas_dso_handle, #__name);         \
      return reinterpret_cast<cublasFunc>(p_##__name)(args...);     \
    }                                                               \
  };                                                                \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)     \
  struct DynLoad__##__name {                         \
    template <typename... Args>                      \
    inline cublasStatus_t operator()(Args... args) { \
      return __name(args...);                        \
    }                                                \
  };                                                 \
  extern DynLoad__##__name __name
#endif

#define DECLARE_DYNAMIC_LOAD_CUBLAS_V2_WRAP(__name) \
  DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)

#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSgemv_v2);                \
  __macro(cublasDgemv_v2);                \
  __macro(cublasSgemm_v2);                \
  __macro(cublasDgemm_v2);                \
  __macro(cublasSgeam_v2);                \
  __macro(cublasDgeam_v2);                \
  __macro(cublasCreate_v2);               \
  __macro(cublasDestroy_v2);              \
  __macro(cublasSetStream_v2);            \
  __macro(cublasSetPointerMode_v2);       \
  __macro(cublasGetPointerMode_v2);       \
  __macro(cublasSgemmBatched);            \
  __macro(cublasDgemmBatched);            \
  __macro(cublasCgemmBatched);            \
  __macro(cublasZgemmBatched);            \
  __macro(cublasSgemmStridedBatched);     \
  __macro(cublasDgemmStridedBatched);     \
  __macro(cublasCgemmStridedBatched);     \
  __macro(cublasZgemmStridedBatched);     \
  __macro(cublasSgetrfBatched);           \
  __macro(cublasSgetriBatched);           \
  __macro(cublasDgetrfBatched);           \
  __macro(cublasDgetriBatched)

CUBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP);

#undef DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
