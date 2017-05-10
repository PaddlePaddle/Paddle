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

#if defined(__CUDA_ARCH__)
#include <vector_types.h>
#ifndef PADDLE_TYPE_DOUBLE
typedef float4 vecType;
#else
typedef double2 vecType;
#endif
#elif (defined  __ARM_NEON) || (defined __ARM_NEON__)
#include <arm_neon.h>
#ifndef PADDLE_TYPE_DOUBLE
typedef float32x4_t  vecType;
#else
#error NEON instructions does not support double precision
#endif
#else
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#ifndef PADDLE_TYPE_DOUBLE
typedef __m128  vecType;
#else
typedef __m128d vecType;
#endif
#endif

#ifdef __CUDA_ARCH__
#define INLINE   __device__ inline
#else
#define INLINE   inline
#endif

#endif  // HL_MATRIX_TYPE_CUH_
