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

#include "paddle/pten/backends/dynload/nvrtc.h"

namespace paddle {
namespace platform {
namespace dynload {

extern bool HasNVRTC();

#define PLATFORM_DECLARE_DYNAMIC_LOAD_NVRTC_WRAP(__name)      \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

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

NVRTC_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_NVRTC_WRAP);

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_NVRTC_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
