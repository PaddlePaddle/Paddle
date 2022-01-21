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
#ifdef PADDLE_WITH_HIP
#include <hipfft.h>

#include <mutex>  // NOLINT

#include "paddle/pten/backends/dynload/dynamic_loader.h"
#include "paddle/pten/backends/dynload/port.h"

namespace pten {
namespace dynload {
extern std::once_flag hipfft_dso_flag;
extern void *hipfft_dso_handle;

#define DECLARE_DYNAMIC_LOAD_HIPFFT_WRAP(__name)                     \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using hipfftFunc = decltype(&::__name);                        \
      std::call_once(hipfft_dso_flag, []() {                         \
        hipfft_dso_handle = pten::dynload::GetROCFFTDsoHandle();     \
      });                                                            \
      static void *p_##__name = dlsym(hipfft_dso_handle, #__name);   \
      return reinterpret_cast<hipfftFunc>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define HIPFFT_FFT_ROUTINE_EACH(__macro) \
  __macro(hipfftPlan1d);                 \
  __macro(hipfftPlan2d);                 \
  __macro(hipfftPlan3d);                 \
  __macro(hipfftPlanMany);               \
  __macro(hipfftMakePlan1d);             \
  __macro(hipfftMakePlanMany);           \
  __macro(hipfftMakePlanMany64);         \
  __macro(hipfftGetSizeMany64);          \
  __macro(hipfftEstimate1d);             \
  __macro(hipfftEstimate2d);             \
  __macro(hipfftEstimate3d);             \
  __macro(hipfftEstimateMany);           \
  __macro(hipfftCreate);                 \
  __macro(hipfftGetSize1d);              \
  __macro(hipfftGetSizeMany);            \
  __macro(hipfftGetSize);                \
  __macro(hipfftSetWorkArea);            \
  __macro(hipfftSetAutoAllocation);      \
  __macro(hipfftExecC2C);                \
  __macro(hipfftExecR2C);                \
  __macro(hipfftExecC2R);                \
  __macro(hipfftExecZ2Z);                \
  __macro(hipfftExecD2Z);                \
  __macro(hipfftExecZ2D);                \
  __macro(hipfftSetStream);              \
  __macro(hipfftDestroy);                \
  __macro(hipfftGetVersion);             \
  __macro(hipfftGetProperty);

HIPFFT_FFT_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HIPFFT_WRAP);

inline const char *hipfftGetErrorString(hipfftResult_t status) {
  switch (status) {
    case HIPFFT_SUCCESS:
      return "'HIPFFT_SUCCESS'. The hipFFT operation was successful.";
    case HIPFFT_INVALID_PLAN:
      return "'HIPFFT_INVALID_PLAN'. hipFFT was passed an invalid plan handle.";
    case HIPFFT_ALLOC_FAILED:
      return "'HIPFFT_ALLOC_FAILED'. hipFFT failed to allocate GPU or CPU "
             "memory.";
    case HIPFFT_INVALID_TYPE:
      return "'HIPFFT_INVALID_TYPE'. No longer used.";
    case HIPFFT_INVALID_VALUE:
      return "'HIPFFT_INVALID_VALUE'. User specified an invalid pointer or "
             "parameter.";
    case HIPFFT_INTERNAL_ERROR:
      return "'HIPFFT_INTERNAL_ERROR'. Driver or internal hipFFT library "
             "error.";
    case HIPFFT_EXEC_FAILED:
      return "'HIPFFT_EXEC_FAILED'. Failed to execute an FFT on the GPU.";
    case HIPFFT_SETUP_FAILED:
      return "'HIPFFT_SETUP_FAILED'. The hipFFT library failed to initialize.";
    case HIPFFT_INVALID_SIZE:
      return "'HIPFFT_INVALID_SIZE'. User specified an invalid transform size.";
    case HIPFFT_UNALIGNED_DATA:
      return "'HIPFFT_UNALIGNED_DATA'. No longer used.";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
      return "'HIPFFT_INCOMPLETE_PARAMETER_LIST'. Missing parameters in call.";
    case HIPFFT_INVALID_DEVICE:
      return "'HIPFFT_INVALID_DEVICE'. Execution of a plan was on different "
             "GPU than plan creation.";
    case HIPFFT_PARSE_ERROR:
      return "'HIPFFT_PARSE_ERROR'. Internal plan database error.";
    case HIPFFT_NO_WORKSPACE:
      return "'HIPFFT_NO_WORKSPACE'. No workspace has been provided prior to "
             "plan execution.";
    case HIPFFT_NOT_IMPLEMENTED:
      return "'HIPFFT_NOT_IMPLEMENTED'. Function does not implement "
             "functionality for parameters given.";
    case HIPFFT_NOT_SUPPORTED:
      return "'HIPFFT_NOT_SUPPORTED'. Operation is not supported for "
             "parameters given.";
    default:
      return "HIPFFT_STATUS_UNKNOWN_ERROR";
  }
}
}  // namespace dynload
}  // namespace pten

#endif
