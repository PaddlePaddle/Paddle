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

#include <mtrtc.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag musartc_dso_flag;
extern void* musartc_dso_handle;
extern bool HasNVRTC();

#define DECLARE_DYNAMIC_LOAD_NVRTC_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using musartc_func = decltype(&::__name);                      \
      std::call_once(musartc_dso_flag, []() {                        \
        musartc_dso_handle = phi::dynload::GetNVRTCDsoHandle();      \
      });                                                            \
      static void* p_##__name = dlsym(musartc_dso_handle, #__name);  \
      return reinterpret_cast<musartc_func>(p_##__name)(args...);    \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed musartc functions
 **/
#define MUSARTC_ROUTINE_EACH(__macro) \
  __macro(mtrtcVersion);              \
  __macro(mtrtcGetErrorString);       \
  __macro(mtrtcCompileProgram);       \
  __macro(mtrtcCreateProgram);        \
  __macro(mtrtcDestroyProgram);       \
  __macro(mtrtcGetMUSA);              \
  __macro(mtrtcGetMUSASize);          \
  __macro(mtrtcGetProgramLog);        \
  __macro(mtrtcGetProgramLogSize)

MUSARTC_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NVRTC_WRAP);

#undef DECLARE_DYNAMIC_LOAD_NVRTC_WRAP

}  // namespace dynload
}  // namespace phi
