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

// need to define PADDLE_ARM_FP16

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>

#include <cuda.h>

#ifdef USE_EIGEN  // delete this #if macro
#include "Eigen/src/Core/arch/CUDA/Half.h"
#endif

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

#ifdef __CUDACC__
#define PADDLE_HOSTDEVICE __host__ __device__
#if CUDA_VERSION >= 7050
#define PADDLE_CUDA_FP16
#include <cuda_fp16.h>
#endif  // CUDA_VERSION >= 7050
#else
#define PADDLE_HOSTDEVICE
#endif  // __CUDACC__

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
#include <arm_neon.h>
#endif

#if defined(PADDLE_NEON) && defined(PADDLE_ARM_32)
#define PADDLE_NEON_32
#endif

#if defined(PADDLE_NEON) && defined(PADDLE_ARM_64)
#define PADDLE_NEON_64
#endif

#ifdef PADDLE_ARM
#ifdef __F16C__
#undef __F16C__
#endif  // __F16C__
#else
#include <immintrin.h>
#endif  // PADDLE_ARM

#define PADDLE_ALIGN(x) __attribute__((aligned(x)))

namespace paddle {

struct float16;

namespace fp16_impl {
// convert from float to half precision in round-to-nearest-even mode
PADDLE_HOSTDEVICE inline float16 float_to_half_rn(float f);
PADDLE_HOSTDEVICE inline float half_to_float(float16 h);
}  // namespace fp16_impl

// Use PADDLE_ALIGNED(2) to ensure that each float16 will be allocated
// and aligned at least on a 2-byte boundary, which leads to efficient
// memory access of float16 struct and also makes float16 compatible
// with CUDA half, ARM float16_t, and Eigen::half data types.
struct PADDLE_ALIGN(2) float16 {
  uint16_t x;

  // explicit for different types, implicit for half and Eigen::half

  PADDLE_HOSTDEVICE inline float16() {}

  PADDLE_HOSTDEVICE inline float16(const float16& h) : x(h.x) {}

#ifdef PADDLE_CUDA_FP16
  PADDLE_HOSTDEVICE inline float16(const half& h) {
#if CUDA_VERSION >= 9000
    x = reinterpret_cast<__half_raw*>(&h)->x;
#else
    x = h.x;
#endif  // CUDA_VERSION >= 9000
  }
#endif  // PADDLE_CUDA_FP16

#ifdef USE_EIGEN
  PADDLE_HOSTDEVICE inline float16(const Eigen::half& h) : x(h.x) {}
#endif  // USE_EIGEN

#ifdef PADDLE_NEON
  // __fp16 is a native half precision data type for arm cpu,
  // float16_t is an alias for __fp16 in arm_fp16.h,
  // which is included in arm_neon.h.
  // According to gcc, __fp16 can only be used as an argument to fp16
  // intrinsic defined in arm_neon.h or as a storage type. It cannot
  // be used as a formal function argument.
  // TODO (kexinzhao): test it on RPI
  PADDLE_HOSTDEVICE inline float16(const float16_t* h) {
    x = *reinterpret_cast<uint16_t*>(h);
  }
#endif

  PADDLE_HOSTDEVICE inline explicit float16(bool b) : x(b ? 0x3c00 : 0) {}

  PADDLE_HOSTDEVICE inline explicit float16(int8_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(uint8_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(int16_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(uint16_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(int32_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(uint32_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(int64_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(uint64_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(float val) {
    float16 res = fp16_impl::float_to_half_rn(val);
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline explicit float16(double val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(const float16& rhs) {
    x = rhs.x;
    return *this;
  }

#ifdef PADDLE_CUDA_FP16
  PADDLE_HOSTDEVICE inline float16& operator=(const half& rhs) {
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

#ifdef PADDLE_NEON
  PADDLE_HOSTDEVICE inline float16& operator=(const float16_t* rhs) {
    x = *reinterpret_cast<uint16_t*>(rhs);
    return *this;
  }
#endif

  PADDLE_HOSTDEVICE inline float16& operator=(bool b) {
    x = b ? 0x3c00 : 0;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(int8_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(uint8_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(int16_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(uint16_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(int32_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(uint32_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(int64_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(uint64_t val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(float val) {
    float16 res = fp16_impl::float_to_half_rn(val);
    x = res.x;
    return *this;
  }

  PADDLE_HOSTDEVICE inline float16& operator=(double val) {
    float16 res = fp16_impl::float_to_half_rn(static_cast<float>(val));
    x = res.x;
    return *this;
  }

#ifdef PADDLE_CUDA_FP16
  PADDLE_HOSTDEVICE inline operator half() const {
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
  PADDLE_HOSTDEVICE inline operator Eigen::half() const {
    Eigen::half h;
    h.x = x;
    return h;
  }
#endif  // USE_EIGEN

#ifdef PADDLE_NEON
  // check whether it works or not
  PADDLE_HOSTDEVICE inline operator float16_t() const {
    float16 h = *this;
    return *reinterpret_cast<float16_t*>(&h);
  }
#endif

  PADDLE_HOSTDEVICE inline explicit operator bool() const {
    return (x & 0x7fff) != 0;
  }

  PADDLE_HOSTDEVICE inline explicit operator int8_t() const {
    return static_cast<int8_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint8_t() const {
    return static_cast<uint8_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator int16_t() const {
    return static_cast<int16_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint16_t() const {
    return static_cast<uint16_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator int32_t() const {
    return static_cast<int32_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint32_t() const {
    return static_cast<uint32_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator int64_t() const {
    return static_cast<int64_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator uint64_t() const {
    return static_cast<uint64_t>(fp16_impl::half_to_float(*this));
  }

  PADDLE_HOSTDEVICE inline explicit operator float() const {
    return fp16_impl::half_to_float(*this);
  }

  PADDLE_HOSTDEVICE inline explicit operator double() const {
    return static_cast<double>(fp16_impl::half_to_float(*this));
  }
};

// arithmetic operators
#if defined(PADDLE_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
__device__ inline float16 operator+(const float16& a, const float16& b) {
  return float16(__hadd(half(a), half(b)));
}

__device__ inline float16 operator-(const float16& a, const float16& b) {
  return float16(__hsub(half(a), half(b)));
}

__device__ inline float16 operator*(const float16& a, const float16& b) {
  return float16(__hmul(half(a), half(b)));
}

__device__ inline float16 operator/(const float16& a, const float16& b) {
  // TODO(kexinzhao): check the cuda version that starts to support __hdiv
  // instinsic
  float num = __half2float(half(a));
  float denom = __half2float(half(b));
  return float16(num / denom);
}

__device__ inline float16 operator-(const float16& a) {
  return float16(__hneg(half(a)));
}

__device__ inline float16& operator+=(float16& a, const float16& b) {
  a = a + b;
  return a;
}

__device__ inline float16& operator-=(float16& a, const float16& b) {
  a = a - b;
  return a;
}

__device__ inline float16& operator*=(float16& a, const float16& b) {
  a = a * b;
  return a;
}

__device__ inline float16& operator/=(float16& a, const float16& b) {
  a = a / b;
  return a;
}

__device__ inline bool operator==(const float16& a, const float16& b) {
  return __heq(half(a), half(b));
}

__device__ inline bool operator!=(const float16& a, const float16& b) {
  return __hne(half(a), half(b));
}

__device__ inline bool operator<(const float16& a, const float16& b) {
  return __hlt(half(a), half(b));
}

__device__ inline bool operator<=(const float16& a, const float16& b) {
  return __hle(half(a), half(b));
}

__device__ inline bool operator>(const float16& a, const float16& b) {
  return __hgt(half(a), half(b));
}

__device__ inline bool operator>=(const float16& a, const float16& b) {
  return __hge(half(a), half(b));
}

// On ARMv8.2-A CPU
#elif (PADDLE_GNUC_VER >= 71 || PADDLE_CLANG_VER >= 39) && \
    defined(PADDLE_NEON_64) && defined(PADDLE_ARM_FP16)
__host__ inline float16 operator+(const float16& a, const float16& b) {
  return float16(vaddh_f16(float16_t(a), float16_t(b)));
}

__host__ inline float16 operator-(const float16& a, const float16& b) {
  return float16(vsubh_f16(float16_t(a), float16_t(b)));
}

__host__ inline float16 operator*(const float16& a, const float16& b) {
  return float16(vmulh_f16(float16_t(a), float16_t(b)));
}

__host__ inline float16 operator/(const float16& a, const float16& b) {
  return float16(vdivh_f16(float16_t(a), float16_t(b)));
}

__host__ inline float16 operator-(const float16& a) {
  return float16(vnegh_f16(float16_t(a)));
}

__host__ inline float16& operator+=(float16& a, const float16& b) {
  a = a + b;
  return a;
}

__host__ inline float16& operator-=(float16& a, const float16& b) {
  a = a - b;
  return a;
}

__host__ inline float16& operator*=(float16& a, const float16& b) {
  a = a * b;
  return a;
}

__host__ inline float16& operator/=(float16& a, const float16& b) {
  a = a / b;
  return a;
}

__host__ inline bool operator==(const float16& a, const float16& b) {
  return static_cast<bool>(vceqh_f16(float16_t(a), float16_t(b)));
}

__host__ inline bool operator!=(const float16& a, const float16& b) {
  return !(a == b);
}

// compare only available in NEON_64
__host__ inline bool operator<(const float16& a, const float16& b) {
  return static_cast<bool>(vclth_f16(float16_t(a), float16_t(b)));
}

__host__ inline bool operator<=(const float16& a, const float16& b) {
  return static_cast<bool>(vcleh_f16(float16_t(a), float16_t(b)));
}

__host__ inline bool operator>(const float16& a, const float16& b) {
  return static_cast<bool>(vcgth_f16(float16_t(a), float16_t(b)));
}

__host__ inline bool operator>=(const float16& a, const float16& b) {
  return static_cast<bool>(vcgeh_f16(float16_t(a), float16_t(b)));
}

#else  // software emulation on other cpu
PADDLE_HOSTDEVICE inline float16 operator+(const float16& a, const float16& b) {
  return float16(float(a) + float(b));
}

PADDLE_HOSTDEVICE inline float16 operator-(const float16& a, const float16& b) {
  return float16(float(a) - float(b));
}

PADDLE_HOSTDEVICE inline float16 operator*(const float16& a, const float16& b) {
  return float16(float(a) * float(b));
}

PADDLE_HOSTDEVICE inline float16 operator/(const float16& a, const float16& b) {
  return float16(float(a) / float(b));
}

PADDLE_HOSTDEVICE inline float16 operator-(const float16& a) {
  float16 res;
  res.x = a.x ^ 0x8000;
  return res;
}

PADDLE_HOSTDEVICE inline float16& operator+=(float16& a, const float16& b) {
  a = float16(float(a) + float(b));
  return a;
}

PADDLE_HOSTDEVICE inline float16& operator-=(float16& a, const float16& b) {
  a = float16(float(a) - float(b));
  return a;
}

PADDLE_HOSTDEVICE inline float16& operator*=(float16& a, const float16& b) {
  a = float16(float(a) * float(b));
  return a;
}

PADDLE_HOSTDEVICE inline float16& operator/=(float16& a, const float16& b) {
  a = float16(float(a) / float(b));
  return a;
}

PADDLE_HOSTDEVICE inline bool operator==(const float16& a, const float16& b) {
  return float(a) == float(b);
}

PADDLE_HOSTDEVICE inline bool operator!=(const float16& a, const float16& b) {
  return float(a) != float(b);
}

PADDLE_HOSTDEVICE inline bool operator<(const float16& a, const float16& b) {
  return float(a) < float(b);
}

PADDLE_HOSTDEVICE inline bool operator<=(const float16& a, const float16& b) {
  return float(a) <= float(b);
}

PADDLE_HOSTDEVICE inline bool operator>(const float16& a, const float16& b) {
  return float(a) > float(b);
}

PADDLE_HOSTDEVICE inline bool operator>=(const float16& a, const float16& b) {
  return float(a) >= float(b);
}

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

#elif defined(PADDLE_NEON_64)  // test on RPI
  float16 res;
  asm volatile(
      "ld1 {v0.s}[0], [%[float_ptr]]\n"
      "fcvt h0, s0\n"
      "st1 {v0.h}[0], [%[half_ptr]]\n"
      :  // outputs
      :  // inputs
      [float_ptr] "r"(&f),
      [half_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "v0");
  return res;

#elif defined(PADDLE_NEON_32)  // test on RPI
  float16 res;
  asm volatile(
      "vld1.32 {d0[0]}, [%[float_ptr]]\n"
      "vcvt.f16.f32 d0, q0\n"
      "vst1.16 {d0[0]}, [%[half_ptr]]\n"
      :  // outputs
      :  // inputs
      [float_ptr] "r"(&f),
      [half_ptr] "r"(&(res.x))
      :  // clobbers
      "memory", "d0");
  return res;

#elif defined(__F16C__)
  float16 res;
  res.x = _cvtss_sh(f, 0);
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

#elif defined(PADDLE_NEON_64)
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

#elif defined(PADDLE_NEON_32)
  float res;
  asm volatile(
      "vld1.16 {d0[0]}, [%[half_ptr]]\n"
      "vcvt.f32.f16 q0, d0\n"
      "vst1.32 {d0[0]}, [%[float_ptr]]\n"
      :  // outputs
      :  // inputs
      [half_ptr] "r"(&(h.x)),
      [float_ptr] "r"(&res)
      :  // clobbers
      "memory", "v0");
  return res;

#elif defined(__F16C__)
  return _cvtsh_ss(h.x);

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

}  // namespace half_impl

}  // namespace paddle
