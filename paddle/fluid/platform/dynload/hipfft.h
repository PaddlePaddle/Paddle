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

#include "paddle/pten/backends/dynload/hipfft.h"

namespace paddle {
namespace platform {
namespace dynload {
extern std::once_flag hipfft_dso_flag;
extern void *hipfft_dso_handle;

#define PLATFORM_DECLARE_DYNAMIC_LOAD_HIPFFT_WRAP(__name)     \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
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

HIPFFT_FFT_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_HIPFFT_WRAP);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif
