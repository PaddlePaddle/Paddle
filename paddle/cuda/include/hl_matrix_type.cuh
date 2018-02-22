/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_MATRIX_TYPE_CUH_
#define HL_MATRIX_TYPE_CUH_

#include "hl_base.h"

#ifdef __CUDA_ARCH__
/**
 * CUDA kernel inline function
 */
#define INLINE   __device__ inline
#else
/**
 * CPP inline function
 */
#define INLINE   inline
#endif

#ifdef __CUDA_ARCH__
#include <vector_types.h>
#ifndef PADDLE_TYPE_DOUBLE
typedef float4 vecType;
#else
typedef double2 vecType;
#endif
#elif defined(__SSE3__)
#include "hl_cpu_simd_sse.cuh"
#define PADDLE_USE_SSE3
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(__NVCC__)
// Currently nvcc does not support neon intrinsic.
// TODO: Extract simd intrinsic implementation from .cu files.
#include "hl_cpu_simd_neon.cuh"
#define PADDLE_USE_NEON
#else
#include "hl_cpu_scalar.cuh"
#endif

#endif  // HL_MATRIX_TYPE_CUH_
