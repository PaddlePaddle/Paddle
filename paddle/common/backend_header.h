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

#ifdef __CUDACC__
#include <cuda.h>
#endif
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) && (CUDA_VERSION >= 11000 || defined(PADDLE_WITH_COREX))
#define PADDLE_CUDA_BF16
#define CINN_CUDA_BF16
#include <cuda_bf16.h>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#define CINN_CUDA_FP16
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#define PADDLE_CUDA_FP16
#define CINN_HIP_FP16
#include <hip/hip_fp16.h>
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

#define CUDA_ARCH_FP16_SUPPORTED(CUDA_ARCH) (CUDA_ARCH >= 600)
