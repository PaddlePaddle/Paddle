/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag musa_dso_flag;
extern void* musa_dso_handle;
extern bool HasCUDADriver();

#define DECLARE_DYNAMIC_LOAD_MUSA_WRAP(__name)                       \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using musa_func = decltype(&::__name);                         \
      std::call_once(musa_dso_flag, []() {                           \
        musa_dso_handle = phi::dynload::GetCUDADsoHandle();          \
      });                                                            \
      static void* p_##__name = dlsym(musa_dso_handle, #__name);     \
      return reinterpret_cast<musa_func>(p_##__name)(args...);       \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed musa driver functions
 **/
#define MUSA_ROUTINE_EACH(__macro)                      \
  __macro(muInit);                                      \
  __macro(muDriverGetVersion);                          \
  __macro(muGetErrorString);                            \
  __macro(muModuleLoadData);                            \
  __macro(muModuleGetFunction);                         \
  __macro(muModuleUnload);                              \
  __macro(muOccupancyMaxActiveBlocksPerMultiprocessor); \
  __macro(muLaunchKernel);                              \
  __macro(muCtxCreate);                                 \
  __macro(muCtxGetCurrent);                             \
  __macro(muDeviceGetCount);                            \
  __macro(muDevicePrimaryCtxGetState);                  \
  __macro(muDeviceGetAttribute);                        \
  __macro(muDeviceGet);

MUSA_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MUSA_WRAP);

#undef DECLARE_DYNAMIC_LOAD_MUSA_WRAP

}  // namespace dynload
}  // namespace phi
