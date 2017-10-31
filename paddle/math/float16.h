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

#include <cuda.h>  // seems need to delete it

#ifdef USE_EIGEN  // delete this #if macro
#include "Eigen/src/Core/arch/CUDA/Half.h"
#endif

#ifdef __CUDACC__
#define PADDLE_HOSTDEVICE __host__ __device__
#if CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif  // CUDA_VERSION >= 7050
#else
#define PADDLE_HOSTDEVICE
#endif  // __CUDA_ARCH__

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

#if defined(PADDLE_ARM) && defined(PADDLE_NEON)
#include <arm_neon.h>
#endif

#if !defined(__ANDROID__) && !defined(__APPLE__) && !defined(PADDLE_ARM)
#include <immintrin.h>
#else
#ifdef __F16C__
#undef __F16C__
#endif
#endif

#define PADDLE_ALIGN(x) __attribute__((aligned(x)))

// https://github.com/pytorch/pytorch/blob/master/torch/lib/ATen/Half.h
template <typename To, typename From>
To convert(From f) {
  return static_cast<To>(f);
}

namespace paddle {

struct float16;

namespace fp16_impl {
// convert from float to half precision in round-to-nearest-even mode
PADDLE_HOSTDEVICE inline float16 float_to_half_rn(float f);
PADDLE_HOSTDEVICE inline float half_to_float(float16 h);
PADDLE_HOSTDEVICE inline float16 uint16_to_half(uint16_t x);
}  // namespace fp16_impl

// Use PADDLE_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with CUDA half and Eigen::half data types.
struct PADDLE_ALIGN(2) float16 {
  uint16_t x;

  // explicit for different types, implicit for half and Eigen::half

  PADDLE_HOSTDEVICE inline float16() {}

  PADDLE_HOSTDEVICE inline float16(const float16& h) : x(h.x) {}

#ifdef PADDLE_CUDA_FP16
  PADDLE_HOSTDEVICE inline float16(const half h) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(&h)->x;
#else
    x = h.x;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // PADDLE_CUDA_FP16
/*
#ifdef PADDLE_CUDA_FP16
  #if CUDA_VERSION < 9000
  PADDLE_HOSTDEVICE inline float16(const half& h) : x(h.x) {}
  #else
  PADDLE_HOSTDEVICE inline float16(const __half_raw& h) : x(h.x) {}
  PADDLE_HOSTDEVICE inline float16(const half& h)
    : x(*reinterpret_cast<uint16_t*>(&h)) {}
  #endif // CUDA_VERSION < 9000
#endif // PADDLE_CUDA_FP16
*/

#ifdef USE_EIGEN
  PADDLE_HOSTDEVICE inline float16(const Eigen::half& h) : x(h.x) {}
#endif  // USE_EIGEN

#if defined(PADDLE_ARM) && defined(PADDLE_NEON)
  // __fp16 is a native half precision data type for arm cpu,
  // float16_t is an alias for __fp16 in arm_fp16.h
  // which is included in arm_neon.h
  PADDLE_HOSTDEVICE inline float16(const float16_t h) {
    x = *reinterpret_cast<uint16_t*>(&h);
  }
#endif

  PADDLE_HOSTDEVICE inline explicit float16(bool b) : x(b ? 0x3c00 : 0) {}

  PADDLE_HOSTDEVICE inline explicit float16(float val) {
    float16 res = fp16_impl::float_to_half_rn(val);
    x = res.x;
  }

  template <class T>
  PADDLE_HOSTDEVICE inline explicit float16(const T& val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(const float16& rhs) {
    x = rhs.x;
    return *this;
  }

#ifdef PADDLE_CUDA_FP16
  PADDLE_HOSTDEVICE inline float16& operator=(const half rhs) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(&rhs)->x;
#else
    x = rhs.x;
#endif
    return *this;
  }
#endif

#ifdef USE_EIGEN
  PADDLE_HOSTDEVICE inline float16& operator=(const Eigen::half& rhs) {
    x = rhs.x;
    return *this;
  }
#endif  // USE_EIGEN

#if defined(PADDLE_ARM) && defined(PADDLE_NEON)
  PADDLE_HOSTDEVICE inline float16& operator=(const float16_t rhs) {
    x = *reinterpret_cast<uint16_t*>(&rhs);
    return *this;
  }
#endif

/*
  PADDLE_HOSTDEVICE inline explicit float16(int val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(double val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }
*/

#ifdef PADDLE_CUDA_FP16
  PADDLE_HOSTDEVICE inline operator half() {
#if CUDA_VERSION >= 9000
    __half_raw h;
    h.x = x;
    return half(h);
#else
    half h;
    h.x = x;
    return h;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // PADDLE_CUDA_FP16

#ifdef USE_EIGEN
  PADDLE_HOSTDEVICE inline operator Eigen::half() {
    Eigen::half h;
    h.x = x;
    return h;
  }
#endif  // USE_EIGEN

#if defined(PADDLE_ARM) && defined(PADDLE_NEON)
  PADDLE_HOSTDEVICE inline operator float16_t() {
    float16 h = *this;
    return *reinterpret_cast<float16_t*>(&h);
  }
#endif

  PADDLE_HOSTDEVICE inline explicit operator bool() {
    return (x & 0x7fff) != 0;
  }

  PADDLE_HOSTDEVICE inline explicit operator int8_t() {
    return static_cat<int8_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint8_t() {
    return static_cat<uint8_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator int16_t() {
    return static_cat<int16_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint16_t() {
    return static_cat<uint16_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator int32_t() {
    return static_cat<int32_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint32_t() {
    return static_cat<uint32_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator int64_t() {
    return static_cat<int64_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint64_t() {
    return static_cat<uint64_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator float() {
    return fp16_impl::half_to_float(*this);
  }

  PADDLE_HOSTDEVICE inline explicit operator double() {
    return static_cat<double>(fp16_impl::half_to_float(*this));
  }
};

// arithmetic operators
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
__device__ inline float16 operator+(const float16& a, const float16& b) {
  return float16(__hadd(a, b));
}

__device__ inline float16 operator-(const float16& a, const float16& b) {
  return __hsub(a, b);
}

__device__ inline float16 operator*(const float16& a, const float16& b) {
  return __hmul(a, b);
}

#elif  // on arm cpu

#else

#endif

namespace fp16_impl {

Union Bits {
  float f;
  int32_t si;
  uint32_t ui;
};

const int shift = 13;
const int shiftSign = 16;

const int32_t infN = 0x7F800000;
const int32_t maxN = 0x477FE000;  // max flt16 as flt32
const int32_t minN = 0x38800000;  // min flt16 normal as flt32
const int32_t sigN = 0x80000000;  // sign bit

constexpr int32_t infC = infN >> shift;
constexpr int32_t nanN = (infC + 1) << shift;  // minimum flt16 nan as float32
constexpr int32_t maxC = maxN >> shift;
constexpr int32_t minC = minN >> shift;
constexpr int32_t sigC = sigN >> shiftSign;

const int32_t mulN = 0x52000000;  //(1 << 23) / minN
const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
const int32_t norC = 0x00400;     // min flt32 normal downshifted

constexpr int32_t maxD = infC - maxC - 1;
constexpr int32_t minD = minC - subC - 1;

PADDLE_HOSTDEVICE inline float16 float_to_half_rn(float f) {
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  half tmp = __float2half(f);
  return *reinterpret_cast<float16*>(&(tmp));

#elif defined(__F16C__)
  float16 res;
  res.x = _cvtss_sh(f, 0);
  return res;

#elif defined(PADDLE_ARM_64)  // test on RPI
  float16 res;
  asm volatile(
      "ld1 {v0.s}[0], [%[float_ptr]]\n"
      "FCVT h0, s0\n"
      "st1 {v0.h}[0], [%[half_ptr]]\n"
      :  // outputs
      :  // inputs
      [float_ptr] "r"(&f),
      [half_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0");
  return res;

#else
  // Conversion routine adapted from
  // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
  Bits v, s;
  v.f = f;
  uint32_t sign = v.si & sigN;
  v.si ^= sign;
  sign >>= shiftSign;  // logical shift
  s.si = mulN;
  s.si = s.f * v.f;  // correct subnormals
  v.si ^= (s.si ^ v.si) & -(minN > v.si);
  v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
  v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
  v.ui >>= shift;  // logical shift
  v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
  v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
  float16 res;
  res.x = v.ui | sign;
  return res;

#endif
}

PADDLE_HOSTDEVICE inline float half_to_float(float16 h) {
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  half tmp = *reinterpret_cast<half*>(&h);
  return __half2float(h);

#elif defined(__F16C__)
  return _cvtsh_ss(h.x);

#elif defined(PADDLE_ARM_64)  // test on RPI
  float res;
  asm volatile(
      "ld1 {v0.h}[0], [%[half_ptr]]\n"
      "FCVT s0, h0\n"
      "st1 {v0.s}[0], [%[float_ptr]]\n"
      :  // outputs
      :  // inputs
      [half_ptr] "r"(&(h.x)),
      [float_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0");
  return res;

#else
  // Conversion routine adapted from
  // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
  Bits v;
  v.ui = x;
  int32_t sign = v.si & sigC;
  v.si ^= sign;
  sign <<= shiftSign;
  v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
  v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
  Bits s;
  s.si = mulC;
  s.f *= v.si;
  int32_t mask = -(norC > v.si);
  v.si <<= shift;
  v.si ^= (s.si ^ v.si) & mask;
  v.si |= sign;
  return v.f;

#endif
}

PADDLE_HOSTDEVICE inline float16 uint16_to_half(uint16_t x) {
  float16 res;
  res.x = x;
  return res;
}

}  // namespace half_impl

}  // namespace paddle
