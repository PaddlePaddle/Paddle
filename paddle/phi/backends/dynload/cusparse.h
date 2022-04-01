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

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {
extern std::once_flag cusparse_dso_flag;
extern void *cusparse_dso_handle;

#define DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP(__name)                   \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    cusparseStatus_t operator()(Args... args) {                      \
      using cusparseFunc = decltype(&::__name);                      \
      std::call_once(cusparse_dso_flag, []() {                       \
        cusparse_dso_handle = phi::dynload::GetCusparseDsoHandle();  \
      });                                                            \
      static void *p_##__name = dlsym(cusparse_dso_handle, #__name); \
      return reinterpret_cast<cusparseFunc>(p_##__name)(args...);    \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#if defined(PADDLE_WITH_CUDA)
// The generic APIs is supported from CUDA10.1
#if CUDA_VERSION >= 10010
#define CUSPARSE_ROUTINE_EACH(__macro) \
  __macro(cusparseCreate);             \
  __macro(cusparseSetStream);          \
  __macro(cusparseCreateMatDescr);     \
  __macro(cusparseDestroy);            \
  __macro(cusparseSnnz);               \
  __macro(cusparseDnnz);               \
  __macro(cusparseSetMatType);         \
  __macro(cusparseSetMatIndexBase);

CUSPARSE_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP);

// APIs available after CUDA 11.2
#if CUDA_VERSION >= 11020
#define CUSPARSE_ROUTINE_EACH_11020(__macro) \
  __macro(cusparseCreateCsr);                \
  __macro(cusparseCreateCoo);                \
  __macro(cusparseCreateDnMat);              \
  __macro(cusparseSpMM_bufferSize);          \
  __macro(cusparseSpMM);                     \
  __macro(cusparseDestroySpMat);             \
  __macro(cusparseDestroyDnMat);             \
  __macro(cusparseCooSetPointers);           \
  __macro(cusparseCsrSetPointers);           \
  __macro(cusparseDenseToSparse_bufferSize); \
  __macro(cusparseDenseToSparse_analysis);   \
  __macro(cusparseDenseToSparse_convert);    \
  __macro(cusparseSparseToDense_bufferSize); \
  __macro(cusparseSparseToDense);

CUSPARSE_ROUTINE_EACH_11020(DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP)

// APIs available after CUDA 11.3
#if CUDA_VERSION >= 11030
#define CUSPARSE_ROUTINE_EACH_R2(__macro) \
  __macro(cusparseSDDMM_bufferSize);      \
  __macro(cusparseSDDMM_preprocess);      \
  __macro(cusparseSDDMM);

CUSPARSE_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP)
#endif
#endif
#endif
#endif

#undef DECLARE_DYNAMIC_LOAD_CUSPARSE_WRAP
}  // namespace dynload
}  // namespace phi
