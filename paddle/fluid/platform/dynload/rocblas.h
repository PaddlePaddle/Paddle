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

#include "paddle/phi/backends/dynload/rocblas.h"

namespace paddle {
namespace platform {
namespace dynload {

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define PLATFORM_DECLARE_DYNAMIC_LOAD_ROCBLAS_WRAP(__name)   \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define ROCBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(rocblas_caxpy);                  \
  __macro(rocblas_saxpy);                  \
  __macro(rocblas_daxpy);                  \
  __macro(rocblas_zaxpy);                  \
  __macro(rocblas_sscal);                  \
  __macro(rocblas_dscal);                  \
  __macro(rocblas_scopy);                  \
  __macro(rocblas_dcopy);                  \
  __macro(rocblas_cgemv);                  \
  __macro(rocblas_sgemv);                  \
  __macro(rocblas_zgemv);                  \
  __macro(rocblas_dgemv);                  \
  __macro(rocblas_cgemm);                  \
  __macro(rocblas_sgemm);                  \
  __macro(rocblas_dgemm);                  \
  __macro(rocblas_hgemm);                  \
  __macro(rocblas_zgemm);                  \
  __macro(rocblas_sgeam);                  \
  __macro(rocblas_strsm);                  \
  __macro(rocblas_dtrsm);                  \
  __macro(rocblas_dgeam);                  \
  __macro(rocblas_sgemm_batched);          \
  __macro(rocblas_dgemm_batched);          \
  __macro(rocblas_cgemm_batched);          \
  __macro(rocblas_zgemm_batched);          \
  __macro(rocblas_create_handle);          \
  __macro(rocblas_destroy_handle);         \
  __macro(rocblas_set_stream);             \
  __macro(rocblas_get_stream);             \
  __macro(rocblas_set_pointer_mode);       \
  __macro(rocblas_get_pointer_mode);

ROCBLAS_BLAS_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_ROCBLAS_WRAP)

// APIs available after CUDA 8.0
#define ROCBLAS_BLAS_ROUTINE_EACH_R2(__macro) \
  __macro(rocblas_gemm_ex);                   \
  __macro(rocblas_sgemm_strided_batched);     \
  __macro(rocblas_dgemm_strided_batched);     \
  __macro(rocblas_cgemm_strided_batched);     \
  __macro(rocblas_zgemm_strided_batched);     \
  __macro(rocblas_hgemm_strided_batched);

ROCBLAS_BLAS_ROUTINE_EACH_R2(PLATFORM_DECLARE_DYNAMIC_LOAD_ROCBLAS_WRAP)

// HIP not supported in ROCM3.5
// #define ROCBLAS_BLAS_ROUTINE_EACH_R3(__macro)
//   __macro(cublasSetMathMode);
//   __macro(cublasGetMathMode);
// ROCBLAS_BLAS_ROUTINE_EACH_R3(PLATFORM_DECLARE_DYNAMIC_LOAD_ROCBLAS_WRAP)

#define ROCBLAS_BLAS_ROUTINE_EACH_R4(__macro) \
  __macro(rocblas_gemm_batched_ex);           \
  __macro(rocblas_gemm_strided_batched_ex);

ROCBLAS_BLAS_ROUTINE_EACH_R4(PLATFORM_DECLARE_DYNAMIC_LOAD_ROCBLAS_WRAP)

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_ROCBLAS_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
