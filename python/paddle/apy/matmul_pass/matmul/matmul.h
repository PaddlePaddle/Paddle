// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <map>
#include <vector>

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(func)                                                      \
  {                                                                           \
    cudaError_t err = func;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                << "] "                                                       \
                << "CUDA error(" << err << "), " << cudaGetErrorString(err)   \
                << " when call " << #func << std::endl;                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#include "cutlass_patch/cuda/cutlass_matmul.cuh"  // NOLINT
#include "math_function.h"                        // NOLINT
#include "profile.h"                              // NOLINT
#endif

#ifdef __HIPCC__
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define CHECK_HIP(func)                                                       \
  {                                                                           \
    hipError_t err = func;                                                    \
    if (err != hipSuccess) {                                                  \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                << "] "                                                       \
                << "HIP error(" << err << "), " << hipGetErrorString(err)     \
                << " when call " << #func << std::endl;                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

#include "cutlass_patch/hip/hytlass_matmul.h"  // NOLINT
#include "math_function.h"                     // NOLINT
#include "profile.h"                           // NOLINT
#endif
