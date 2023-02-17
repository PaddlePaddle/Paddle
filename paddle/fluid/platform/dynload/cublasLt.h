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

#include "paddle/phi/backends/dynload/cublasLt.h"

namespace paddle {
namespace platform {
namespace dynload {

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublasLt routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define PLATFORM_DECLARE_DYNAMIC_LOAD_CUBLASLT_WRAP(__name)  \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

// APIs available after CUDA 10.1
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

CUBLASLT_BLAS_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_CUBLASLT_WRAP)
// #endif

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_CUBLASLT_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle