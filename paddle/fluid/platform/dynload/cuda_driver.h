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

#include <cuda.h>
#include <mutex>  // NOLINT

#include "paddle/pten/backends/dynload/cuda_driver.h"

namespace paddle {
namespace platform {
namespace dynload {

extern bool HasCUDADriver();

#define PLATFORM_DECLARE_DYNAMIC_LOAD_CUDA_WRAP(__name)       \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

/**
 * include all needed cuda driver functions
 **/
#define CUDA_ROUTINE_EACH(__macro)                      \
  __macro(cuInit);                                      \
  __macro(cuDriverGetVersion);                          \
  __macro(cuGetErrorString);                            \
  __macro(cuModuleLoadData);                            \
  __macro(cuModuleGetFunction);                         \
  __macro(cuModuleUnload);                              \
  __macro(cuOccupancyMaxActiveBlocksPerMultiprocessor); \
  __macro(cuLaunchKernel);                              \
  __macro(cuCtxCreate);                                 \
  __macro(cuCtxGetCurrent);                             \
  __macro(cuDeviceGetCount);                            \
  __macro(cuDevicePrimaryCtxGetState);                  \
  __macro(cuDeviceGetAttribute);                        \
  __macro(cuDeviceGet)

#if CUDA_VERSION >= 10020
#define CUDA_ROUTINE_EACH_VVM(__macro)    \
  __macro(cuMemGetAllocationGranularity); \
  __macro(cuMemAddressReserve);           \
  __macro(cuMemCreate);                   \
  __macro(cuMemMap);                      \
  __macro(cuMemSetAccess);                \
  __macro(cuMemUnmap);                    \
  __macro(cuMemRelease);                  \
  __macro(cuMemAddressFree)

CUDA_ROUTINE_EACH_VVM(PLATFORM_DECLARE_DYNAMIC_LOAD_CUDA_WRAP);
#endif

CUDA_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_CUDA_WRAP);

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_CUDA_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
