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

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

#include <cuda.h>

#ifdef __arm__
#define PADDLE_ARM_32
#endif

#ifdef __aarch64__
#define PADDLE_ARM_64
#endif

#if defined(PADDLE_ARM_32) || defined(PADDLE_ARM_64)
#define PADDLE_ARM
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define PADDLE_NEON
#endif

#if defined(PADDLE_NEON) && defined(PADDLE_ARM_32)
#define PADDLE_NEON_32
#endif

#if defined(PADDLE_NEON) && defined(PADDLE_ARM_64)
#define PADDLE_NEON_64
#endif

#ifdef __CUDA_ARCH__  // use __CUDACC__ instead
#define PADDLE_HOSTDEVICE __host__ __device__
#if CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif  // CUDA_VERSION >= 7050
#else
#define PADDLE_HOSTDEVICE
#endif  // __CUDA_ARCH__

#if !defined(__ANDROID__) && !defined(__APPLE__) && !defined(PADDLE_ARM)
#include <immintrin.h>
#else
#ifdef __F16C__
#undef __F16C__
#endif
#endif

#define PADDLE_ALIGNED(x) __attribute__((aligned(x)))

// https://github.com/pytorch/pytorch/blob/master/torch/lib/ATen/Half.h
template <typename To, typename From>
To convert(From f) {
  return static_cast<To>(f);
}

namespace paddle {

class float16;

// convert from float to half precision in round-to-nearest-even mode
float16 float2half_rn(float f);
float half2float(float16 h);

class float16 {
public:
  uint16_t val_;

  PADDLE_HOSTDEVICE inline explicit float16() : x(0) {}

  PADDLE_HOSTDEVICE inline explicit float16(float val) {
    float16 res = float2half_rn(val);
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(int val) {
    float16 res = cpu_float2half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(double val) {
    float16 res = cpu_float2half_rn(static_cast<float>(val));
    x = res.x;
  }

  // Use PADDLE_ALIGNED(2) to ensure that each float16 will be allocated
  // and aligned at least on a 2-byte boundary, which leads to efficient
  // memory access of float16 struct.
} PADDLE_ALIGNED(2);

namespace fp16_impl {

// Conversion routine adapted from
// http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
Union Bits {
  float f;
  int32_t si;
  uint32_t ui;
};

static const int shift = 13;
static const int shiftSign = 16;

static const int32_t infN = 0x7F800000;
static const int32_t maxN = 0x477FE000;  // max flt16 as flt32
static const int32_t minN = 0x38800000;  // min flt16 normal as flt32
static const int32_t sigN = 0x80000000;  // sign bit

static constexpr int32_t infC = infN >> shift;
static constexpr int32_t nanN = (infC + 1)
                                << shift;  // minimum flt16 nan as float32
static constexpr int32_t maxC = maxN >> shift;
static constexpr int32_t minC = minN >> shift;
static constexpr int32_t sigC = sigN >> shiftSign;

static const int32_t mulN = 0x52000000;  //(1 << 23) / minN
static const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
static const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
static const int32_t norC = 0x00400;     // min flt32 normal downshifted

static constexpr int32_t maxD = infC - maxC - 1;
static constexpr int32_t minD = minC - subC - 1;

}  // namespace half_impl

}  // namespace paddle
