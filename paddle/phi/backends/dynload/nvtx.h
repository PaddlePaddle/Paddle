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
#ifndef _WIN32
#include <cuda.h>
#ifndef NVTX_SUPPRESS_V2_DEPRECATION_WARNING
#define NVTX_SUPPRESS_V2_DEPRECATION_WARNING
#endif

#if (CUDA_VERSION >= 12080) || defined(PADDLE_WITH_XPU)
// CCCL 3.1.0 is enabled for CUDA >= 12.8 (see cmake/external/cccl.cmake) and
// pulls in the NVTX3 headers through thrust. Use the NVTX3-compatible entry
// point over the same CUDA range so this header does not also include the
// legacy <nvToolsExt.h>; the two use different include guards and defining both
// leads to duplicate NVTX type definitions (nvtxStringHandle_t,
// nvtxEventAttributes_v2, ...). Keep this boundary in sync with the CCCL gate.
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

#ifdef PADDLE_WITH_XPU
#define DECLARE_DIRECT_NVTX_WRAP(__name)                    \
  struct DynLoad__##__name {                                \
    template <typename... Args>                             \
    int operator()(Args... args) {                          \
      using nvtxFunc = decltype(&::__name);                 \
      return reinterpret_cast<nvtxFunc>(::__name)(args...); \
    }                                                       \
  };                                                        \
  extern DynLoad__##__name __name

#define NVTX_ROUTINE_EACH(__macro) \
  __macro(nvtxRangePushA);         \
  __macro(nvtxRangePushEx);        \
  __macro(nvtxRangePop);

NVTX_ROUTINE_EACH(DECLARE_DIRECT_NVTX_WRAP);

#undef DECLARE_DIRECT_NVTX_WRAP
#else
extern std::once_flag nvtx_dso_flag;
extern void *nvtx_dso_handle;

#define DECLARE_DYNAMIC_LOAD_NVTX_WRAP(__name)                   \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    int operator()(Args... args) {                               \
      using nvtxFunc = decltype(&::__name);                      \
      std::call_once(nvtx_dso_flag, []() {                       \
        nvtx_dso_handle = phi::dynload::GetNvtxDsoHandle();      \
      });                                                        \
      static void *p_##__name = dlsym(nvtx_dso_handle, #__name); \
      return reinterpret_cast<nvtxFunc>(p_##__name)(args...);    \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name

#define NVTX_ROUTINE_EACH(__macro) \
  __macro(nvtxRangePushA);         \
  __macro(nvtxRangePushEx);        \
  __macro(nvtxRangePop);

NVTX_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NVTX_WRAP);

#undef DECLARE_DYNAMIC_LOAD_NVTX_WRAP
#endif

}  // namespace dynload
}  // namespace phi
#endif
