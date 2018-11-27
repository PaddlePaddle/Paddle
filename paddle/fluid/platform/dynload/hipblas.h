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

#include <hipblas.h>
#include <dlfcn.h>
#include <mutex>
#include <type_traits>
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag hipblas_dso_flag;
extern void *hipblas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP(__name)                             \
  struct DynLoad__##__name {                                                 \
    using FUNC_TYPE = decltype(&::__name);                                   \
    template <typename... Args>                                              \
    inline hipblasStatus_t operator()(Args... args) {                         \
      std::call_once(hipblas_dso_flag, []() {                                 \
        hipblas_dso_handle = paddle::platform::dynload::GetHipblasDsoHandle(); \
      });                                                                    \
      void *p_##__name = dlsym(hipblas_dso_handle, #__name);                  \
      return reinterpret_cast<FUNC_TYPE>(p_##__name)(args...);               \
    }                                                                        \
  };                                                                         \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP(__name)     \
  struct DynLoad__##__name {                         \
    template <typename... Args>                      \
    inline hipblasStatus_t operator()(Args... args) { \
      return __name(args...);                        \
    }                                                \
  };                                                 \
  extern DynLoad__##__name __name
#endif

#define HIPBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(hipblasSaxpy);                \
  __macro(hipblasDaxpy);                \
  __macro(hipblasSgemv);                \
  __macro(hipblasDgemv);                \
  __macro(hipblasSgemm);                \
  __macro(hipblasDgemm);                \
  __macro(hipblasHgemm);                \
  __macro(hipblasSgeam);                \
  __macro(hipblasDgeam);                \
  __macro(hipblasCreate);               \
  __macro(hipblasDestroy);              \
  __macro(hipblasSetStream);            \
  __macro(hipblasSetPointerMode);       \
  __macro(hipblasGetPointerMode);       \
  __macro(hipblasSgemmBatched);            \
  __macro(hipblasDgemmBatched);            \
  __macro(hipblasSgemmStridedBatched);     \
  __macro(hipblasDgemmStridedBatched);

HIPBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP);

#undef DECLARE_DYNAMIC_LOAD_HIPBLAS_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
