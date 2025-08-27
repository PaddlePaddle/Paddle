/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

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
#ifdef PADDLE_WITH_XPU_FFT
#include <fft/cufft.h>
#include <glog/logging.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag cufft_dso_flag;
extern void* cufft_dso_handle;
extern bool HasCUFFT();

extern void EnforceCUFFTLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_CUFFT_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using cufft_func = decltype(&::__name);                        \
      std::call_once(cufft_dso_flag, []() {                          \
        cufft_dso_handle = phi::dynload::GetCUFFTDsoHandle();        \
      });                                                            \
      EnforceCUFFTLoaded(#__name);                                   \
      static void* p_##__name = dlsym(cufft_dso_handle, #__name);    \
      return reinterpret_cast<cufft_func>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed cufft functions in HPPL
 * different cufft version has different interfaces
 **/
#define CUFFT_FFT_ROUTINE_EACH(__macro) \
  __macro(cufftPlan1d);                 \
  __macro(cufftPlan2d);                 \
  __macro(cufftPlan3d);                 \
  __macro(cufftPlanMany);               \
  __macro(cufftMakePlan1d);             \
  __macro(cufftMakePlan2d);             \
  __macro(cufftMakePlan3d);             \
  __macro(cufftMakePlanMany);           \
  __macro(cufftEstimate1d);             \
  __macro(cufftEstimate2d);             \
  __macro(cufftEstimate3d);             \
  __macro(cufftEstimateMany);           \
  __macro(cufftCreate);                 \
  __macro(cufftGetSize1d);              \
  __macro(cufftGetSize2d);              \
  __macro(cufftGetSize3d);              \
  __macro(cufftGetSizeMany);            \
  __macro(cufftGetSize);                \
  __macro(cufftSetWorkArea);            \
  __macro(cufftSetAutoAllocation);      \
  __macro(cufftExecC2C);                \
  __macro(cufftExecR2C);                \
  __macro(cufftExecC2R);                \
  __macro(cufftSetStream);              \
  __macro(cufftDestroy);                \
  __macro(cufftGetVersion);             \
  __macro(cufftGetProperty);

CUFFT_FFT_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUFFT_WRAP)

}  // namespace dynload
}  // namespace phi
#endif
