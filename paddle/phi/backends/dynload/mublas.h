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

#include <mublas.h>
#include <musa.h>

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag mublas_dso_flag;
extern void *mublas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load mublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP(__name)                            \
  struct DynLoad__##__name {                                                \
    template <typename... Args>                                             \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using mublas_func =                                                   \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);         \
      std::call_once(mublas_dso_flag, []() {                                \
        mublas_dso_handle = phi::dynload::GetCublasDsoHandle();             \
      });                                                                   \
      static void *p_##__name = dlsym(mublas_dso_handle, #__name);          \
      return reinterpret_cast<mublas_func>(p_##__name)(args...);            \
    }                                                                       \
  };                                                                        \
  extern DynLoad__##__name __name

#define MUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(mublasSaxpy);                \
  __macro(mublasDaxpy);                \
  __macro(mublasCaxpy);                \
  __macro(mublasZaxpy);                \
  __macro(mublasSscal);                \
  __macro(mublasDscal);                \
  __macro(mublasScopy);                \
  __macro(mublasDcopy);                \
  __macro(mublasSgemv);                \
  __macro(mublasDgemv);                \
  __macro(mublasCgemv);                \
  __macro(mublasZgemv);                \
  __macro(mublasSgemm);                \
  __macro(mublasDgemm);                \
  __macro(mublasCgemm);                \
  __macro(mublasZgemm);                \
  __macro(mublasHgemm);                \
  __macro(mublasSgeam);                \
  __macro(mublasDgeam);                \
  __macro(mublasDtrsm);                \
  __macro(mublasCtrsm);                \
  __macro(mublasZtrsm);                \
  __macro(mublasCreate);               \
  __macro(mublasDestroy);              \
  __macro(mublasSetStream);            \
  __macro(mublasSetPointerMode);       \
  __macro(mublasGetPointerMode);       \
  __macro(mublasSgemmBatched);         \
  __macro(mublasDgemmBatched);         \
  __macro(mublasCgemmBatched);         \
  __macro(mublasZgemmBatched);


MUBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP)

#undef DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP
}  // namespace dynload
}  // namespace phi
