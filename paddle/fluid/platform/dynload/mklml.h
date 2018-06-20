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

#include <dlfcn.h>
#include <mkl.h>
#include <mutex>  // NOLINT
#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag mklml_dso_flag;
extern void* mklml_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load mklml routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_MKLML_WRAP(__name)                                    \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    auto operator()(Args... args) -> decltype(__name(args...)) {           \
      using mklmlFunc = decltype(&::__name);                               \
      std::call_once(mklml_dso_flag, []() {                                \
        mklml_dso_handle = paddle::platform::dynload::GetMKLMLDsoHandle(); \
      });                                                                  \
      static void* p_##_name = dlsym(mklml_dso_handle, #__name);           \
      return reinterpret_cast<mklmlFunc>(p_##_name)(args...);              \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_MKLML_WRAP(__name) DYNAMIC_LOAD_MKLML_WRAP(__name)

#define MKLML_ROUTINE_EACH(__macro) \
  __macro(cblas_sgemm);             \
  __macro(cblas_saxpy);             \
  __macro(cblas_scopy);             \
  __macro(cblas_sgemv);             \
  __macro(cblas_sgemm_batch);       \
  __macro(cblas_dgemm);             \
  __macro(cblas_daxpy);             \
  __macro(cblas_dcopy);             \
  __macro(cblas_dgemv);             \
  __macro(cblas_dgemm_batch);       \
  __macro(vsAdd);                   \
  __macro(vdAdd);                   \
  __macro(MKL_Set_Num_Threads)

MKLML_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MKLML_WRAP);

#undef DYNAMIC_LOAD_MKLML_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
