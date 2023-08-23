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

#include <musa.h>
#include <musparse.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {
extern std::once_flag musparse_dso_flag;
extern void *musparse_dso_handle;

#define DECLARE_DYNAMIC_LOAD_MUSPARSE_WRAP(__name)                   \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    musparseStatus_t operator()(Args... args) {                      \
      using Func = decltype(&::__name);                              \
      std::call_once(musparse_dso_flag, []() {                       \
        musparse_dso_handle = phi::dynload::GetCusparseDsoHandle();  \
      });                                                            \
      static void *p_##__name = dlsym(musparse_dso_handle, #__name); \
      return reinterpret_cast<Func>(p_##__name)(args...);            \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#if defined(PADDLE_WITH_MUSA)
#define MUSPARSE_ROUTINE_EACH(__macro)   \
  __macro(musparseSetStream);            \
  __macro(musparseCreateMatDescr);       \
  __macro(musparseSnnz);                 \
  __macro(musparseDnnz);                 \
  __macro(musparseSetMatType);           \
  __macro(musparseSetMatIndexBase);      \
  __macro(musparseCreateCsr);            \
  __macro(musparseCreateCoo);            \
  __macro(musparseCreateDnMat);          \
  __macro(musparseCreateDnVec);          \
  __macro(musparseSpMM);                 \
  __macro(musparseDestroySpMat);         \
  __macro(musparseDestroyDnMat);         \
  __macro(musparseDestroyDnVec);         \
  __macro(musparseSpMV);                 \
  __macro(musparseSDDMM_bufferSize);     \
  __macro(musparseSDDMM_preprocess);     \
  __macro(musparseSDDMM);                \
  __macro(musparseDnMatSetStridedBatch); \
  __macro(musparseCooSetStridedBatch);   \
  __macro(musparseCsrSetStridedBatch);

MUSPARSE_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MUSPARSE_WRAP)

#endif  // PADDLE_WITH_MUSA

#undef DECLARE_DYNAMIC_LOAD_MUSPARSE_WRAP
}  // namespace dynload
}  // namespace phi
