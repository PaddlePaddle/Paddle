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
#include <nvToolsExt.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {
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
}  // namespace dynload
}  // namespace phi
#endif
