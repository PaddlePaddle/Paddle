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

std::once_flag cublas_dso_flag;
void *cublas_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DYNAMIC_LOAD_CUBLAS_WRAP(__name)                            \
  struct DynLoad__##__name {                                        \
    template <typename... Args>                                     \
    cublasStatus_t operator()(Args... args) {                       \
      typedef cublasStatus_t (*cublasFunc)(Args...);                \
      std::call_once(cublas_dso_flag,                               \
                     paddle::platform::dynload::GetCublasDsoHandle, \
                     &cublas_dso_handle);                           \
      void *p_##__name = dlsym(cublas_dso_handle, #__name);         \
      return reinterpret_cast<cublasFunc>(p_##__name)(args...);     \
    }                                                               \
  } __name;  // struct DynLoad__##__name
#else
#define DYNAMIC_LOAD_CUBLAS_WRAP(__name)      \
  struct DynLoad__##__name {                  \
    template <typename... Args>               \
    cublasStatus_t operator()(Args... args) { \
      return __name(args...);                 \
    }                                         \
  } __name;  // struct DynLoad__##__name
#endif

#define DYNAMIC_LOAD_CUBLAS_V2_WRAP(__name) DYNAMIC_LOAD_CUBLAS_WRAP(__name)

// include all needed cublas functions in HPPL
// clang-format off
#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSgemv)                    \
  __macro(cublasDgemv)                    \
  __macro(cublasSgemm)                    \
  __macro(cublasDgemm)                    \
  __macro(cublasSgeam)                    \
  __macro(cublasDgeam)                    \

DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasCreate)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasDestroy)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasSetStream)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasSetPointerMode)
DYNAMIC_LOAD_CUBLAS_V2_WRAP(cublasGetPointerMode)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasCgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasZgemmBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgetrfBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasSgetriBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgetrfBatched)
DYNAMIC_LOAD_CUBLAS_WRAP(cublasDgetriBatched)
CUBLAS_BLAS_ROUTINE_EACH(DYNAMIC_LOAD_CUBLAS_V2_WRAP)

#undef DYNAMIC_LOAD_CUBLAS_WRAP
#undef DYNAMIC_LOAD_CUBLAS_V2_WRAP
#undef CUBLAS_BLAS_ROUTINE_EACH

// clang-format on
#ifndef PADDLE_TYPE_DOUBLE
#define CUBLAS_GEAM paddle::platform::dynload::cublasSgeam
#define CUBLAS_GEMV paddle::platform::dynload::cublasSgemv
#define CUBLAS_GEMM paddle::platform::dynload::cublasSgemm
#define CUBLAS_GETRF paddle::platform::dynload::cublasSgetrfBatched
#define CUBLAS_GETRI paddle::platform::dynload::cublasSgetriBatched
#else
#define CUBLAS_GEAM paddle::platform::dynload::cublasDgeam
#define CUBLAS_GEMV paddle::platform::dynload::cublasDgemv
#define CUBLAS_GEMM paddle::platform::dynload::cublasDgemm
#define CUBLAS_GETRF paddle::platform::dynload::cublasDgetrfBatched
#define CUBLAS_GETRI paddle::platform::dynload::cublasDgetriBatched
#endif
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
