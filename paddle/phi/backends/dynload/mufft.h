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
#ifdef PADDLE_WITH_MUSA
#include <mufft.h>
#include <mufftXt.h>
#include <glog/logging.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag mufft_dso_flag;
extern void* mufft_dso_handle;

extern void EnforceMUFFTLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_MUFFT_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using mufft_func = decltype(&::__name);                        \
      std::call_once(mufft_dso_flag, []() {                          \
        mufft_dso_handle = phi::dynload::GetMUFFTDsoHandle();        \
      });                                                            \
      EnforceMUFFTLoaded(#__name);                                   \
      static void* p_##__name = dlsym(mufft_dso_handle, #__name);    \
      return reinterpret_cast<mufft_func>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed mufft functions in HPPL
 * different mufft version has different interfaces
 **/
#define MUFFT_FFT_ROUTINE_EACH(__macro)  \
  __macro(mufftPlan1d);                  \
  __macro(mufftPlan2d);                  \
  __macro(mufftPlan3d);                  \
  __macro(mufftPlanMany);                \
  __macro(mufftMakePlan1d);              \
  __macro(mufftMakePlan2d);              \
  __macro(mufftMakePlan3d);              \
  __macro(mufftMakePlanMany);            \
  __macro(mufftEstimate1d);              \
  __macro(mufftEstimate2d);              \
  __macro(mufftEstimate3d);              \
  __macro(mufftEstimateMany);            \
  __macro(mufftCreate);                  \
  __macro(mufftGetSize1d);               \
  __macro(mufftGetSize2d);               \
  __macro(mufftGetSize3d);               \
  __macro(mufftGetSizeMany);             \
  __macro(mufftGetSize);                 \
  __macro(mufftSetWorkArea);             \
  __macro(mufftSetAutoAllocation);       \
  __macro(mufftExecC2C);                 \
  __macro(mufftExecR2C);                 \
  __macro(mufftExecC2R);                 \
  __macro(mufftExecZ2Z);                 \
  __macro(mufftExecD2Z);                 \
  __macro(mufftExecZ2D);                 \
  __macro(mufftSetStream);               \
  __macro(mufftDestroy);                 \
  __macro(mufftGetVersion);              \
  __macro(mufftGetProperty);             \
  __macro(mufftXtSetGPUs);               \
  __macro(mufftXtMalloc);                \
  __macro(mufftXtMemcpy);                \
  __macro(mufftXtFree);                  \
  __macro(mufftXtExecDescriptorC2C);     \
  __macro(mufftXtExecDescriptorR2C);     \
  __macro(mufftXtExecDescriptorC2R);     \
  __macro(mufftXtExecDescriptorZ2Z);     \
  __macro(mufftXtExecDescriptorD2Z);     \
  __macro(mufftXtExecDescriptorZ2D);     \
  __macro(mufftXtQueryPlan);             \
  __macro(mufftXtSetCallback);           \
  __macro(mufftXtClearCallback);         \
  __macro(mufftXtMakePlanMany);          \
  __macro(mufftXtGetSizeMany);           \
  __macro(mufftXtExec);                  \
  __macro(mufftXtExecDescriptor);        
MUFFT_FFT_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MUFFT_WRAP)


inline const char *mufftGetErrorString(mufftResult_t status) {
  switch (status) {
    case MUFFT_SUCCESS:
      return "'MUFFT_SUCCESS'. The mufft operation was successful.";
    case MUFFT_INVALID_PLAN:
      return "'MUFFT_INVALID_PLAN'. mufft was passed an invalid plan handle.";
    case MUFFT_ALLOC_FAILED:
      return "'MUFFT_ALLOC_FAILED'. mufft failed to allocate GPU or CPU "
             "memory.";
    case MUFFT_INVALID_TYPE:
      return "'MUFFT_INVALID_TYPE'. No longer used.";
    case MUFFT_INVALID_VALUE:
      return "'MUFFT_INVALID_VALUE'. User specified an invalid pointer or "
             "parameter.";
    case MUFFT_INTERNAL_ERROR:
      return "'MUFFT_INTERNAL_ERROR'. Driver or internal mufft library "
             "error.";
    case MUFFT_EXEC_FAILED:
      return "'MUFFT_EXEC_FAILED'. Failed to execute an FFT on the GPU.";
    case MUFFT_SETUP_FAILED:
      return "'MUFFT_SETUP_FAILED'. The mufft library failed to initialize.";
    case MUFFT_INVALID_SIZE:
      return "'MUFFT_INVALID_SIZE'. User specified an invalid transform size.";
    case MUFFT_UNALIGNED_DATA:
      return "'MUFFT_UNALIGNED_DATA'. No longer used.";
    case MUFFT_INCOMPLETE_PARAMETER_LIST:
      return "'MUFFT_INCOMPLETE_PARAMETER_LIST'. Missing parameters in call.";
    case MUFFT_INVALID_DEVICE:
      return "'MUFFT_INVALID_DEVICE'. Execution of a plan was on different "
             "GPU than plan creation.";
    case MUFFT_PARSE_ERROR:
      return "'MUFFT_PARSE_ERROR'. Internal plan database error.";
    case MUFFT_NO_WORKSPACE:
      return "'MUFFT_NO_WORKSPACE'. No workspace has been provided prior to "
             "plan execution.";
    case MUFFT_NOT_IMPLEMENTED:
      return "'MUFFT_NOT_IMPLEMENTED'. Function does not implement "
             "functionality for parameters given.";
    case MUFFT_LICENSE_ERROR:
      return "'MUFFT_LICENSE_ERROR'. Operation is not supported for "
             "parameters given.";
    case MUFFT_NOT_SUPPORTED:
      return "'MUFFT_NOT_SUPPORTED'. Operation is not supported for "
             "parameters given.";                 
    default:
      return "mufft_STATUS_UNKNOWN_ERROR";
  }
}

}  // namespace dynload
}  // namespace phi

#endif
