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

#ifdef PADDLE_WITH_CUDA
#include <nvjpeg.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {
extern std::once_flag nvjpeg_dso_flag;
extern void *nvjpeg_dso_handle;

#define DECLARE_DYNAMIC_LOAD_NVJPEG_WRAP(__name)                   \
  struct DynLoad__##__name {                                       \
    template <typename... Args>                                    \
    nvjpegStatus_t operator()(Args... args) {                      \
      using nvjpegFunc = decltype(&::__name);                      \
      std::call_once(nvjpeg_dso_flag, []() {                       \
        nvjpeg_dso_handle = phi::dynload::GetNvjpegDsoHandle();    \
      });                                                          \
      static void *p_##__name = dlsym(nvjpeg_dso_handle, #__name); \
      return reinterpret_cast<nvjpegFunc>(p_##__name)(args...);    \
    }                                                              \
  };                                                               \
  extern DynLoad__##__name __name

#define NVJPEG_RAND_ROUTINE_EACH(__macro) \
  __macro(nvjpegCreateSimple);            \
  __macro(nvjpegJpegStateCreate);         \
  __macro(nvjpegGetImageInfo);            \
  __macro(nvjpegJpegStateDestroy);        \
  __macro(nvjpegDecode);

NVJPEG_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NVJPEG_WRAP);

}  // namespace dynload
}  // namespace phi

#endif
