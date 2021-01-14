/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <cusolverDn.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {
extern std::once_flag cusolver_dso_flag;
extern void *cusolver_dso_handle;

#define DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP(__name)                   \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    cusolverStatus_t operator()(Args... args) {                      \
      using cusolverFunc = decltype(&::__name);                      \
      std::call_once(cusolver_dso_flag, []() {                       \
        cusolver_dso_handle =                                        \
            paddle::platform::dynload::GetCusolverDsoHandle();       \
      });                                                            \
      static void *p_##__name = dlsym(cusolver_dso_handle, #__name); \
      return reinterpret_cast<cusolverFunc>(p_##__name)(args...);    \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define CUSOLVER_ROUTINE_EACH(__macro)  \
  __macro(cusolverDnCreate);            \
  __macro(cusolverDnDestroy);           \
  __macro(cusolverDnSetStream);         \
  __macro(cusolverDnSpotrf_bufferSize); \
  __macro(cusolverDnDpotrf_bufferSize); \
  __macro(cusolverDnSpotrf);            \
  __macro(cusolverDnDpotrf);

CUSOLVER_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP);

#if CUDA_VERSION >= 9020
#define CUSOLVER_ROUTINE_EACH_R1(__macro) \
  __macro(cusolverDnSpotrfBatched);       \
  __macro(cusolverDnDpotrfBatched);

CUSOLVER_ROUTINE_EACH_R1(DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP)
#endif

#undef DECLARE_DYNAMIC_LOAD_CUSOLVER_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
