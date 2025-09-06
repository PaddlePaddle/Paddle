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

#include <stdint.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include "paddle/common/hostdevice.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 12080
#define PADDLE_CUDA_FP4
#include <cuda_fp4.h>
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

namespace phi {
namespace dtype {

struct PADDLE_ALIGN(1) float4_e2m1fn_x2 {
 public:
  uint8_t x;

  // Constructors
  float4_e2m1fn_x2() = default;
  HOSTDEVICE inline float4_e2m1fn_x2(uint8_t val) : x(val) {}
  ~float4_e2m1fn_x2() = default;
};

}  // namespace dtype
}  // namespace phi
