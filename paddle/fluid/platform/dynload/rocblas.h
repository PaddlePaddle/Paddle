/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag rocblas_dso_flag;
extern void *rocblas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)                              \
  struct DynLoad__##__name {                                                  \
    template <typename... Args>                                               \
    inline auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {   \
      using rocblas_func =                                                    \
          decltype(::__name(std::declval<Args>()...)) (*)(Args...);           \
      std::call_once(rocblas_dso_flag, []() {                                 \
        rocblas_dso_handle = paddle::platform::dynload::GetCublasDsoHandle(); \
      });                                                                     \
      static void *p_##__name = dlsym(rocblas_dso_handle, #__name);           \
      return reinterpret_cast<rocblas_func>(p_##__name)(args...);             \
    }                                                                         \
  };                                                                          \
  extern DynLoad__##__name __name

#define ROCBLAS_BLAS_ROUTINE_EACH(__macro)            \
  __macro(rocblas_saxpy);                             \
  __macro(rocblas_daxpy);                             \
  __macro(rocblas_sscal);                             \
  __macro(rocblas_dscal);                             \
  __macro(rocblas_scopy);                             \
  __macro(rocblas_dcopy);                             \
  __macro(rocblas_sgemv);                             \
  __macro(rocblas_dgemv);                             \
  __macro(rocblas_sgemm);                             \
  __macro(rocblas_dgemm);                             \
  __macro(rocblas_hgemm);                             \
  __macro(rocblas_dgeam);                             \
  /*rocblas_gemm_ex function not support at rocm3.5*/ \
  /*__macro(rocblas_gemm_ex);                 */      \
  __macro(rocblas_sgemm_batched);                     \
  __macro(rocblas_dgemm_batched);                     \
  __macro(rocblas_cgemm_batched);                     \
  __macro(rocblas_zgemm_batched);                     \
  __macro(rocblas_create_handle);                     \
  __macro(rocblas_destroy_handle);                    \
  __macro(rocblas_add_stream);                        \
  __macro(rocblas_set_stream);                        \
  __macro(rocblas_get_stream);                        \
  __macro(rocblas_set_pointer_mode);                  \
  __macro(rocblas_get_pointer_mode);

ROCBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

#define ROCBLAS_BLAS_ROUTINE_EACH_R2(__macro) \
  __macro(rocblas_sgemm_strided_batched);     \
  __macro(rocblas_dgemm_strided_batched);     \
  __macro(rocblas_cgemm_strided_batched);     \
  __macro(rocblas_zgemm_strided_batched);     \
  __macro(rocblas_hgemm_strided_batched);

ROCBLAS_BLAS_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

#define ROCBLAS_BLAS_ROUTINE_EACH_R3(__macro)

ROCBLAS_BLAS_ROUTINE_EACH_R3(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

#define ROCBLAS_BLAS_ROUTINE_EACH_R4(__macro) \
  __macro(rocblas_gemm_batched_ex);           \
// rocm not support now(rocm3.5)
//  __macro(rocblas_gemm_strided_batched_ex);

ROCBLAS_BLAS_ROUTINE_EACH_R4(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP)

#undef DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
