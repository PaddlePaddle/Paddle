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

#include <nvrtc.h>
#include <mutex>  // NOLINT

#include "paddle/pten/backends/dynload/dynamic_loader.h"
#include "paddle/pten/backends/dynload/port.h"

namespace pten {
namespace dynload {

extern std::once_flag nvrtc_dso_flag;
extern void* nvrtc_dso_handle;
extern bool HasNVRTC();

#define DECLARE_DYNAMIC_LOAD_NVRTC_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using nvrtc_func = decltype(&::__name);                        \
      std::call_once(nvrtc_dso_flag, []() {                          \
        nvrtc_dso_handle = pten::dynload::GetNVRTCDsoHandle();       \
      });                                                            \
      static void* p_##__name = dlsym(nvrtc_dso_handle, #__name);    \
      return reinterpret_cast<nvrtc_func>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed nvrtc functions
 **/
#define NVRTC_ROUTINE_EACH(__macro) \
  __macro(nvrtcVersion);            \
  __macro(nvrtcGetErrorString);     \
  __macro(nvrtcCompileProgram);     \
  __macro(nvrtcCreateProgram);      \
  __macro(nvrtcDestroyProgram);     \
  __macro(nvrtcGetPTX);             \
  __macro(nvrtcGetPTXSize);         \
  __macro(nvrtcGetProgramLog);      \
  __macro(nvrtcGetProgramLogSize)

NVRTC_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NVRTC_WRAP);

#undef DECLARE_DYNAMIC_LOAD_NVRTC_WRAP

}  // namespace dynload
}  // namespace pten
