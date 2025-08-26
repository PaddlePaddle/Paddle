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
  uint8_t _x;

  // Constructors
  float4_e2m1fn_x2() = default;
  HOSTDEVICE inline float4_e2m1fn_x2(uint8_t val) : _x(val) {}
  ~float4_e2m1fn_x2() = default;

  //   HOSTDEVICE inline explicit operator int8_t() const {
  //     return static_cast<int8_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator uint8_t() const {
  //     return static_cast<uint8_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator int16_t() const {
  //     return static_cast<int16_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator uint16_t() const {
  //     return static_cast<uint16_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator int32_t() const {
  //     return static_cast<int32_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator uint32_t() const {
  //     return static_cast<uint32_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator int64_t() const {
  //     return static_cast<int64_t>(static_cast<float>(*this));
  //   }

  //   HOSTDEVICE inline explicit operator uint64_t() const {
  //     return static_cast<uint64_t>(static_cast<float>(*this));
  //   }
};

}  // namespace dtype
}  // namespace phi
