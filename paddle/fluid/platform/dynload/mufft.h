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

#include "paddle/phi/backends/dynload/mufft.h"

namespace paddle {
namespace platform {
namespace dynload {


#define PLATFORM_DECLARE_DYNAMIC_LOAD_MUFFT_WRAP(__name)     \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

/**
 * include all needed cufft functions in HPPL
 * different cufft version has different interfaces
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

MUFFT_FFT_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_MUFFT_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif
