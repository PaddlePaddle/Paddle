/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/dynload/musa_driver.h"

namespace paddle {
namespace platform {
namespace dynload {

extern bool HasCUDADriver();

#define PLATFORM_DECLARE_DYNAMIC_LOAD_MUSA_WRAP(__name)      \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

/**
 * include all needed musa driver functions
 **/
#define PLATFORM_MUSA_ROUTINE_EACH(__macro)             \
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
  __macro(muDeviceGet)

PLATFORM_MUSA_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_MUSA_WRAP);

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_MUSA_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

