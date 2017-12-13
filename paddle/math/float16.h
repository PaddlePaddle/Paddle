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

#include <stdint.h>

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#endif  // PADDLE_WITH_CUDA

#include "unsupported/Eigen/CXX11/Tensor"

#include "paddle/platform/hostdevice.h"

#ifdef __GNUC__
#define PADDLE_GNUC_VER (__GNUC__ * 10 + __GNUC_MINOR__)
#else
#define PADDLE_GNUC_VER 0
#endif  // __GNUC__

#ifdef __clang__
#define PADDLE_CLANG_VER (__clang_major__ * 10 + __clang_minor__)
#else
#define PADDLE_CLANG_VER 0
#endif  // __clang__

#if defined(__HIPCC__) && CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#include <hip/hip_fp16.h>
#endif

#if defined(__arm__) || defined(__aarch64__)
#define PADDLE_ARM
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define PADDLE_NEON
#include <arm_neon.h>
#endif

#if defined(PADDLE_NEON) && defined(PADDLE_ARM_FP16) && \
    (PADDLE_GNUC_VER >= 62 || PADDLE_CLANG_VER >= 37)
#define PADDLE_WITH_NATIVE_FP16
#endif

#ifndef PADDLE_ARM
#include <immintrin.h>
#endif  // PADDLE_ARM

#define PADDLE_ALIGN(x) __attribute__((aligned(x)))

namespace paddle {

// Use PADDLE_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with CUDA half, ARM float16_t, and Eigen::half data types.
struct PADDLE_ALIGN(2) float16 {
public:
  uint16_t x;

  // Constructors
  HOSTDEVICE inline float16() : x(0) {}

  HOSTDEVICE inline float16(const float16& h) : x(h.x) {}

#ifdef PADDLE_CUDA_FP16
  HOSTDEVICE inline explicit float16(const half& h) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(&h)->x;
#else
    x = h.x;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // PADDLE_CUDA_FP16

  HOSTDEVICE inline explicit float16(const Eigen::half& h) : x(h.x) {}

#ifdef PADDLE_WITH_NATIVE_FP16
  // __fp16 is a native half precision data type for arm cpu,
  // float16_t is an alias for __fp16
  HOSTDEVICE inline explicit float16(const float16_t& h) {
    x = *reinterpret_cast<const uint16_t*>(&h);
  }
#endif

  HOSTDEVICE inline explicit float16(float val) {
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    half tmp = __float2half(val);
    x = *reinterpret_cast<uint16_t*>(&tmp);

#elif defined(PADDLE_NEON)
    float32x4_t tmp = vld1q_dup_f32(&val);
    float16_t res = vget_lane_f16(vcvt_f16_f32(tmp), 0);
    x = *reinterpret_cast<uint16_t*>(&res);

#elif defined(__F16C__)
    x = _cvtss_sh(val, 0);

#else
    // Conversion routine adapted from
    // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    Bits v, s;
    v.f = val;
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
    x = v.ui | sign;

#endif
  }

  HOSTDEVICE inline explicit float16(bool b) : x(b ? 0x3c00 : 0) {}

  template <class T>
  HOSTDEVICE inline explicit float16(const T& val)
      : x(float16(static_cast<float>(val)).x) {}

  HOSTDEVICE inline float16& operator=(const float16& rhs) {
    x = rhs.x;
    return *this;
  }

// Assignment operators
#ifdef PADDLE_CUDA_FP16
  HOSTDEVICE inline float16& operator=(const half& rhs) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(&rhs)->x;
#else
    x = rhs.x;
#endif
    return *this;
  }
#endif

  HOSTDEVICE inline float16& operator=(const Eigen::half& rhs) {
    x = rhs.x;
    return *this;
  }

#ifdef PADDLE_WITH_NATIVE_FP16
  HOSTDEVICE inline float16& operator=(const float16_t& rhs) {
    x = *reinterpret_cast<const uint16_t*>(&rhs);
    return *this;
  }
#endif

  HOSTDEVICE inline float16& operator=(bool b) {
    x = b ? 0x3c00 : 0;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int8_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint8_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int16_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint16_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int32_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint32_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(int64_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(uint64_t val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(float val) {
    x = float16(val).x;
    return *this;
  }

  HOSTDEVICE inline float16& operator=(double val) {
    x = float16(val).x;
    return *this;
  }

// Conversion opertors
#ifdef PADDLE_CUDA_FP16
  HOSTDEVICE inline explicit operator half() const {
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

  HOSTDEVICE inline explicit operator Eigen::half() const {
    Eigen::half h;
    h.x = x;
    return h;
  }

#ifdef PADDLE_WITH_NATIVE_FP16
  HOSTDEVICE inline explicit operator float16_t() const {
    return *reinterpret_cast<const float16_t*>(this);
  }
#endif

  HOSTDEVICE inline explicit operator float() const {
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    half tmp = *reinterpret_cast<const half*>(this);
    return __half2float(tmp);

#elif defined(PADDLE_NEON)
    float16x4_t res = vld1_dup_f16(reinterpret_cast<const float16_t*>(this));
    return vgetq_lane_f32(vcvt_f32_f16(res), 0);

#elif defined(__F16C__)
    return _cvtsh_ss(this->x);

#else
    // Conversion routine adapted from
    // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    Bits v;
    v.ui = this->x;
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

  HOSTDEVICE inline explicit operator bool() const { return (x & 0x7fff) != 0; }

  HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(float(*this));
  }

  HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(float(*this));
  }

private:
  union Bits {
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

  static const int32_t mulN = 0x52000000;  // (1 << 23) / minN
  static const int32_t mulC = 0x33800000;  // minN / (1 << (23 - shift))
  static const int32_t subC = 0x003FF;     // max flt32 subnormal downshifted
  static const int32_t norC = 0x00400;     // min flt32 normal downshifted

  static constexpr int32_t maxD = infC - maxC - 1;
  static constexpr int32_t minD = minC - subC - 1;
};

// Arithmetic operators on GPU
// CUDA 9.0 provides built-in arithmetic operators for half while
// CUDA 7.5 and 8.0 do not. The arithmetic operators defined here are
// for users to write similar CUDA code in CUDA 7.5 and 8.0 as in
// CUDA 9.0 regarding the half data type.
#if defined(PADDLE_CUDA_FP16) && CUDA_VERSION < 9000

DEVICE inline half operator+(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hadd(a, b);
#else
  float res = float(float16(a)) + float(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator-(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hsub(a, b);
#else
  float res = float(float16(a)) - float(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator*(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hmul(a, b);
#else
  float res = float(float16(a)) * float(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator/(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
#else
  float res = float(float16(a)) / float(float16(b));
  return half(float16(res));
#endif
}

DEVICE inline half operator-(const half& a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hneg(a);
#else
  float res = -float(float16(a));
  return half(float16(res));
#endif
}

DEVICE inline half& operator+=(half& a, const half& b) {
  a = a + b;
  return a;
}

DEVICE inline half& operator-=(half& a, const half& b) {
  a = a - b;
  return a;
}

DEVICE inline half& operator*=(half& a, const half& b) {
  a = a * b;
  return a;
}

DEVICE inline half& operator/=(half& a, const half& b) {
  a = a / b;
  return a;
}

DEVICE inline bool operator==(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __heq(a, b);
#else
  return float(float16(a)) == float(float16(b));
#endif
}

DEVICE inline bool operator!=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hne(a, b);
#else
  return float(float16(a)) != float(float16(b));
#endif
}

DEVICE inline bool operator<(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b);
#else
  return float(float16(a)) < float(float16(b));
#endif
}

DEVICE inline bool operator<=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hle(a, b);
#else
  return float(float16(a)) <= float(float16(b));
#endif
}

DEVICE inline bool operator>(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hgt(a, b);
#else
  return float(float16(a)) > float(float16(b));
#endif
}

DEVICE inline bool operator>=(const half& a, const half& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hge(a, b);
#else
  return float(float16(a)) >= float(float16(b));
#endif
}

#endif  // PADDLE_CUDA_FP16

// Arithmetic operators on ARMv8.2-A CPU
#if defined(PADDLE_WITH_NATIVE_FP16)
HOST inline float16 operator+(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fadd h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

HOST inline float16 operator-(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fsub h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

HOST inline float16 operator*(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fmul h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

HOST inline float16 operator/(const float16& a, const float16& b) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fdiv h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0", "v1");
  return res;
}

HOST inline float16 operator-(const float16& a) {
  float16 res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "fneg h0, h0\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [res_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0");
  return res;
}

HOST inline float16& operator+=(float16& a, const float16& b) {
  a = a + b;
  return a;
}

HOST inline float16& operator-=(float16& a, const float16& b) {
  a = a - b;
  return a;
}

HOST inline float16& operator*=(float16& a, const float16& b) {
  a = a * b;
  return a;
}

HOST inline float16& operator/=(float16& a, const float16& b) {
  a = a / b;
  return a;
}

HOST inline bool operator==(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fcmeq h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

HOST inline bool operator!=(const float16& a, const float16& b) {
  return !(a == b);
}

HOST inline bool operator<(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v1.h}[0], [%[a_ptr]]\n"
      "ld1 {v0.h}[0], [%[b_ptr]]\n"
      "fcmgt h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

HOST inline bool operator<=(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v1.h}[0], [%[a_ptr]]\n"
      "ld1 {v0.h}[0], [%[b_ptr]]\n"
      "fcmge h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

HOST inline bool operator>(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fcmgt h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

HOST inline bool operator>=(const float16& a, const float16& b) {
  uint16_t res;
  asm volatile(
      "ld1 {v0.h}[0], [%[a_ptr]]\n"
      "ld1 {v1.h}[0], [%[b_ptr]]\n"
      "fcmge h0, h0, h1\n"
      "st1 {v0.h}[0], [%[res_ptr]]\n"
      :  // outputs
      :  // inputs
      [a_ptr] "r"(&(a.x)),
      [b_ptr] "r"(&(b.x)),
      [res_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0", "v1");
  return (res & 0xffff) != 0;
}

// Arithmetic operators, software emulated on other CPU
#else
HOSTDEVICE inline float16 operator+(const float16& a, const float16& b) {
  return float16(float(a) + float(b));
}

HOSTDEVICE inline float16 operator-(const float16& a, const float16& b) {
  return float16(float(a) - float(b));
}

HOSTDEVICE inline float16 operator*(const float16& a, const float16& b) {
  return float16(float(a) * float(b));
}

HOSTDEVICE inline float16 operator/(const float16& a, const float16& b) {
  return float16(float(a) / float(b));
}

HOSTDEVICE inline float16 operator-(const float16& a) {
  float16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

HOSTDEVICE inline float16& operator+=(float16& a, const float16& b) {
  a = float16(float(a) + float(b));
  return a;
}

HOSTDEVICE inline float16& operator-=(float16& a, const float16& b) {
  a = float16(float(a) - float(b));
  return a;
}

HOSTDEVICE inline float16& operator*=(float16& a, const float16& b) {
  a = float16(float(a) * float(b));
  return a;
}

HOSTDEVICE inline float16& operator/=(float16& a, const float16& b) {
  a = float16(float(a) / float(b));
  return a;
}

HOSTDEVICE inline bool operator==(const float16& a, const float16& b) {
  return float(a) == float(b);
}

HOSTDEVICE inline bool operator!=(const float16& a, const float16& b) {
  return float(a) != float(b);
}

HOSTDEVICE inline bool operator<(const float16& a, const float16& b) {
  return float(a) < float(b);
}

HOSTDEVICE inline bool operator<=(const float16& a, const float16& b) {
  return float(a) <= float(b);
}

HOSTDEVICE inline bool operator>(const float16& a, const float16& b) {
  return float(a) > float(b);
}

HOSTDEVICE inline bool operator>=(const float16& a, const float16& b) {
  return float(a) >= float(b);
}
#endif
}  // namespace paddle
