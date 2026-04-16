// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// ROCm HIP public headers require exactly **one** of __HIP_PLATFORM_AMD__ or
// __HIP_PLATFORM_NVIDIA__. Some toolchains/flags leave both undefined, define
// one to 0, or (buggy) define both — all trip `#include <hip/hip_runtime.h>`.
// Paddle ROCm builds target AMD only: normalize to AMD + legacy HCC flags.
#if defined(__HIPCC__) || defined(__HIP__)
#  undef __HIP_PLATFORM_AMD__
#  undef __HIP_PLATFORM_NVIDIA__
#  define __HIP_PLATFORM_AMD__ 1
#  undef __HIP_PLATFORM_HCC__
#  undef __HIP_PLATFORM_NVCC__
#  define __HIP_PLATFORM_HCC__ 1
#endif
