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

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag cuda_dso_flag;
extern void* cuda_dso_handle;
extern bool HasCUDADriver();

#define DECLARE_DYNAMIC_LOAD_CUDA_WRAP(__name)                           \
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {     \
      using cuda_func = decltype(&::__name);                             \
      std::call_once(cuda_dso_flag, []() {                               \
        cuda_dso_handle = paddle::platform::dynload::GetCUDADsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(cuda_dso_handle, #__name);         \
      return reinterpret_cast<cuda_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern struct DynLoad__##__name __name

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
  __macro(cuDevicePrimaryCtxGetState)

CUDA_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUDA_WRAP);

#undef DECLARE_DYNAMIC_LOAD_CUDA_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
