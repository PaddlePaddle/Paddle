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

#include <hip/hip_runtime.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag rocm_dso_flag;
extern void* rocm_dso_handle;
extern bool HasCUDADriver();

#define DECLARE_DYNAMIC_LOAD_ROCM_WRAP(__name)                       \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using rocm_func = decltype(&::__name);                         \
      std::call_once(rocm_dso_flag, []() {                           \
        rocm_dso_handle = phi::dynload::GetCUDADsoHandle();          \
      });                                                            \
      static void* p_##__name = dlsym(rocm_dso_handle, #__name);     \
      return reinterpret_cast<rocm_func>(p_##__name)(args...);       \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed cuda driver functions
 **/
#define ROCM_ROUTINE_EACH(__macro)                            \
  __macro(hipDriverGetVersion);                               \
  __macro(hipGetErrorString);                                 \
  __macro(hipModuleLoadData);                                 \
  __macro(hipModuleGetFunction);                              \
  __macro(hipModuleUnload);                                   \
  /* DTK not support the function*/                           \
  /* __macro(hipOccupancyMaxActiveBlocksPerMultiprocessor);*/ \
  __macro(hipModuleLaunchKernel);                             \
  __macro(hipLaunchKernel);                                   \
  __macro(hipGetDevice);                                      \
  __macro(hipGetDeviceCount);                                 \
  __macro(hipDevicePrimaryCtxGetState);                       \
  __macro(hipDeviceGetAttribute);                             \
  __macro(hipDeviceGet)

#define ROCM_ROUTINE_EACH_VVM(__macro)     \
  __macro(hipMemGetAllocationGranularity); \
  __macro(hipMemAddressReserve);           \
  __macro(hipMemCreate);                   \
  __macro(hipMemMap);                      \
  __macro(hipMemSetAccess);                \
  __macro(hipMemUnmap);                    \
  __macro(hipMemRelease);                  \
  __macro(hipMemAddressFree)

#define ROCM_ROUTINE_EACH_GPU_GRAPH(__macro) \
  __macro(hipGraphNodeGetType);              \
  __macro(hipGraphKernelNodeGetParams);      \
  __macro(hipGraphExecKernelNodeSetParams)

ROCM_ROUTINE_EACH_VVM(DECLARE_DYNAMIC_LOAD_ROCM_WRAP);
ROCM_ROUTINE_EACH_GPU_GRAPH(DECLARE_DYNAMIC_LOAD_ROCM_WRAP);

ROCM_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ROCM_WRAP);

#undef DECLARE_DYNAMIC_LOAD_ROCM_WRAP

}  // namespace dynload
}  // namespace phi
