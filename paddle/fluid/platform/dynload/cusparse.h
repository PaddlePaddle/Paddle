/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <cusparse.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/cusparse.h"

namespace paddle {
namespace platform {
namespace dynload {

#define PLATFORM_DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP(__name)  \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#if defined(PADDLE_WITH_CUDA)
#if CUDA_VERSION >= 11000
#define CUSPARSE_ROUTINE_EACH(__macro) \
  __macro(cusparseCreate);             \
  __macro(cusparseSetStream);          \
  __macro(cusparseCreateMatDescr);     \
  __macro(cusparseDestroy);            \
  __macro(cusparseSnnz);               \
  __macro(cusparseDnnz);               \
  __macro(cusparseSetMatType);         \
  __macro(cusparseSetMatIndexBase);    \
  __macro(cusparseCreateCsr);          \
  __macro(cusparseCreateCoo);          \
  __macro(cusparseCreateDnMat);        \
  __macro(cusparseCreateDnVec);        \
  __macro(cusparseSpMM_bufferSize);    \
  __macro(cusparseSpMM);               \
  __macro(cusparseDestroySpMat);       \
  __macro(cusparseDestroyDnMat);       \
  __macro(cusparseDestroyDnVec);       \
  __macro(cusparseSpMV_bufferSize);    \
  __macro(cusparseSpMV);

CUSPARSE_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP)
#endif

#if CUDA_VERSION >= 11030
#define CUSPARSE_ROUTINE_EACH_R2(__macro) \
  __macro(cusparseSpMM_preprocess);       \
  __macro(cusparseSDDMM_bufferSize);      \
  __macro(cusparseSDDMM_preprocess);      \
  __macro(cusparseSDDMM);

CUSPARSE_ROUTINE_EACH_R2(PLATFORM_DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP)
#endif

#if CUDA_VERSION >= 11070
#define CUSPARSE_ROUTINE_EACH_R3(__macro) \
  __macro(cusparseDnMatSetStridedBatch);  \
  __macro(cusparseCooSetStridedBatch);    \
  __macro(cusparseCsrSetStridedBatch);

CUSPARSE_ROUTINE_EACH_R3(PLATFORM_DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP)
#endif

#endif  // PADDLE_WITH_CUDA

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
