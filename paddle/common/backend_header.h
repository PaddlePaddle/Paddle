// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA)
#include <cuda.h>
#endif

// HIP before CUDA: some HIP toolchains / shims may also define CUDA-ish tokens;
// bf16 must come from HIP headers whenever we are in a HIP compilation unit.
#if defined(__HIPCC__) || defined(__HIP__)
// Provide a CUDA-BF16 compatible surface for HIP compilation units.
// Many GPU kernels use __nv_bfloat16* and conversion intrinsics guarded by
// PADDLE_CUDA_BF16. ROCm/HIP provides equivalent types in hip_bfloat16.h.
#define PADDLE_CUDA_BF16
// hip_bf16.h provides __hip_bfloat16 and __hip_bfloat162 (vector bf16).
// Some ROCm versions also ship hip_bfloat16.h (C++ wrapper types).
#if __has_include(<hip/hip_bf16.h>)
#include <hip/hip_bf16.h>
#endif
#if __has_include(<hip/hip_bfloat16.h>)
#include <hip/hip_bfloat16.h>
#endif
#ifndef __nv_bfloat16
#define __nv_bfloat16 __hip_bfloat16
#endif
#ifndef __nv_bfloat162
#define __nv_bfloat162 __hip_bfloat162
#endif
#elif defined(__CUDACC__)
#define PADDLE_CUDA_BF16
#include <cuda_bf16.h>
#endif

#ifndef PADDLE_WITH_HIP
#if !defined(_WIN32)
#define PADDLE_ALIGN(x) __attribute__((aligned(x)))
#else
#define PADDLE_ALIGN(x) __declspec(align(x))
#endif
#else
#define PADDLE_ALIGN(x)
#endif
