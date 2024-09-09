// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Functor used by mmha kernel.
*/

#pragma once

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef PADDLE_WITH_HIP
#include <float.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#else
#include <cuda_fp16.h>
#include <float.h>
#include <cub/cub.cuh>
#endif

#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#ifdef PADDLE_WITH_HIP
/// integral_constant
template <typename _Tp, _Tp __v>
struct kernel_dtype_integral_constant {
  static constexpr _Tp value = __v;
  typedef _Tp value_type;
  typedef kernel_dtype_integral_constant<_Tp, __v> type;
  constexpr operator value_type() const noexcept { return value; }
};

/// The type used as a compile-time boolean with true value.
typedef kernel_dtype_integral_constant<bool, true> true_type;
/// The type used as a compile-time boolean with false value.
typedef kernel_dtype_integral_constant<bool, false> false_type;

/// is_same
template <typename, typename>
struct kernel_dtype_is_same : public false_type {};

template <typename _Tp>
struct kernel_dtype_is_same<_Tp, _Tp> : public true_type {};

#endif

namespace phi {
namespace fusion {

#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#define MMHA_USE_FP32_ACUM_FOR_OUT
#define MMHA_USE_FP32_ACUM_FOR_FMA

enum CacheType {
  NORMAL,
  INT8,
  INT4,
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

struct Float4_ {
  float2 x;
  float2 y;
};

#if defined(ENABLE_BF16) || defined(PADDLE_WITH_HIP)
struct bf16_4_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};

struct bf16_8_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
};
#endif

//-----------------------------------
template <typename T, CacheType CACHE_TYPE>
struct Packed_Int8_;
template <>
struct Packed_Int8_<float, CacheType::NORMAL> {
  using Type = uint8_t;
};
template <>
struct Packed_Int8_<float2, CacheType::NORMAL> {
  using Type = uint16_t;
};
template <>
struct Packed_Int8_<float4, CacheType::NORMAL> {
  using Type = uint32_t;
};
template <>
struct Packed_Int8_<float, CacheType::INT8> {
  using Type = uint8_t;
};
template <>
struct Packed_Int8_<float2, CacheType::INT8> {
  using Type = uint16_t;
};
template <>
struct Packed_Int8_<float4, CacheType::INT8> {
  using Type = uint32_t;
};

template <>
struct Packed_Int8_<uint32_t, CacheType::NORMAL> {
  using Type = uint16_t;
};
template <>
struct Packed_Int8_<uint2, CacheType::NORMAL> {
  using Type = uint32_t;
};
template <>
struct Packed_Int8_<uint4, CacheType::NORMAL> {
  using Type = uint64_t;
};
template <>
struct Packed_Int8_<uint32_t, CacheType::INT8> {
  using Type = uint16_t;
};
template <>
struct Packed_Int8_<uint2, CacheType::INT8> {
  using Type = uint32_t;
};
template <>
struct Packed_Int8_<uint4, CacheType::INT8> {
  using Type = uint64_t;
};

#ifdef ENABLE_BF16
template <>
struct Packed_Int8_<__nv_bfloat162, CacheType::NORMAL> {
  using Type = uint16_t;
};
template <>
struct Packed_Int8_<bf16_4_t, CacheType::NORMAL> {
  using Type = uint32_t;
};
template <>
struct Packed_Int8_<bf16_8_t, CacheType::NORMAL> {
  using Type = uint64_t;
};
template <>
struct Packed_Int8_<__nv_bfloat162, CacheType::INT8> {
  using Type = uint16_t;
};
template <>
struct Packed_Int8_<bf16_4_t, CacheType::INT8> {
  using Type = uint32_t;
};
template <>
struct Packed_Int8_<bf16_8_t, CacheType::INT8> {
  using Type = uint64_t;
};
#endif

//------------------------------------
template <typename T>
struct num_elems;
template <>
struct num_elems<float> {
  static constexpr int value = 1;
};
template <>
struct num_elems<float2> {
  static constexpr int value = 2;
};
template <>
struct num_elems<float4> {
  static constexpr int value = 4;
};
template <>
struct num_elems<uint32_t> {
  static constexpr int value = 2;
};
template <>
struct num_elems<uint2> {
  static constexpr int value = 4;
};
template <>
struct num_elems<uint4> {
  static constexpr int value = 8;
};
#ifdef ENABLE_BF16
template <>
struct num_elems<__nv_bfloat162> {
  static constexpr int value = 2;
};
template <>
struct num_elems<bf16_4_t> {
  static constexpr int value = 4;
};
template <>
struct num_elems<bf16_8_t> {
  static constexpr int value = 8;
};
#endif

//------------------------------------
template <typename T, int N>
struct packed_type;
template <typename T>
struct packed_type<T, 1> {
  using type = T;
};
template <>
struct packed_type<uint8_t, 2> {
  using type = uint16_t;
};
template <>
struct packed_type<uint8_t, 4> {
  using type = uint32_t;
};
template <>
struct packed_type<uint8_t, 8> {
  using type = uint64_t;
};
template <>
struct packed_type<float, 2> {
  using type = float2;
};
template <>
struct packed_type<float, 4> {
  using type = float4;
};
template <>
struct packed_type<float, 8> {
  using type = Float8_;
};

//------------------------------------
template <typename T, int Dh>
struct Qk_vec_ {};
template <>
struct Qk_vec_<float, 32> {
  using Type = float;
};
template <>
struct Qk_vec_<float, 64> {
  using Type = float2;
};
template <>
struct Qk_vec_<float, 128> {
  using Type = float4;
};
template <>
struct Qk_vec_<float, 256> {
  using Type = float4;
};
template <>
struct Qk_vec_<float16, 32> {
  using Type = uint32_t;
};
template <>
struct Qk_vec_<float16, 64> {
  using Type = uint32_t;
};
template <>
struct Qk_vec_<float16, 128> {
  using Type = uint2;
};
template <>
struct Qk_vec_<float16, 256> {
  using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct Qk_vec_<bfloat16, 32> {
  using Type = __nv_bfloat162;
};
template <>
struct Qk_vec_<bfloat16, 64> {
  using Type = __nv_bfloat162;
};
template <>
struct Qk_vec_<bfloat16, 128> {
  using Type = bf16_4_t;
};
template <>
struct Qk_vec_<bfloat16, 256> {
  using Type = bf16_8_t;
};
#endif  // ENABLE_BF16

// RoPE Type
template <typename T1, typename T2, int Dh>
struct Qk_vec_RoPE_ {};
template <>
struct Qk_vec_RoPE_<float16, float, 32> {
  using Type = float2;
};
template <>
struct Qk_vec_RoPE_<float16, float, 64> {
  using Type = float2;
};
template <>
struct Qk_vec_RoPE_<float16, float, 128> {
  using Type = float4;
};
template <>
struct Qk_vec_RoPE_<float16, float, 256> {
  using Type = Float8_;
};
template <>
struct Qk_vec_RoPE_<float, float, 32> {
  using Type = float;
};
template <>
struct Qk_vec_RoPE_<float, float, 64> {
  using Type = float2;
};
template <>
struct Qk_vec_RoPE_<float, float, 128> {
  using Type = float4;
};
template <>
struct Qk_vec_RoPE_<float, float, 256> {
  using Type = float4;
};
#ifdef ENABLE_BF16
template <>
struct Qk_vec_RoPE_<bfloat16, float, 32> {
  using Type = float2;
};
template <>
struct Qk_vec_RoPE_<bfloat16, float, 64> {
  using Type = float2;
};
template <>
struct Qk_vec_RoPE_<bfloat16, float, 128> {
  using Type = float4;
};
template <>
struct Qk_vec_RoPE_<bfloat16, float, 256> {
  using Type = Float8_;
};
#endif
//------------------------------------

template <typename T, int THREADS_PER_KEY>
struct K_vec_ {};
template <typename T>
struct K_vec_bttn_ {
  using Type = T;
};
template <typename T, CacheType CACHE_TYPE>
struct K_vec_I_bttn_ {
  using Type = uint8_t;
};

template <>
struct K_vec_bttn_<float> {
  using Type = float4;
};
template <>
struct K_vec_bttn_<float16> {
  using Type = uint4;
};

template <>
struct K_vec_I_bttn_<float, CacheType::NORMAL> {
  using Type = uint32_t;
};
template <>
struct K_vec_I_bttn_<float, CacheType::INT8> {
  using Type = uint32_t;
};
template <>
struct K_vec_I_bttn_<float16, CacheType::NORMAL> {
  using Type = uint64_t;
};
template <>
struct K_vec_I_bttn_<float16, CacheType::INT8> {
  using Type = uint64_t;
};

template <>
struct K_vec_<float, 4> {
  using Type = float;
};
template <>
struct K_vec_<float, 2> {
  using Type = float2;
};
template <>
struct K_vec_<float, 1> {
  using Type = float4;
};
template <>
struct K_vec_<float16, 4> {
  using Type = uint32_t;
};
template <>
struct K_vec_<float16, 2> {
  using Type = uint2;
};
template <>
struct K_vec_<float16, 1> {
  using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct K_vec_<bfloat16, 4> {
  using Type = __nv_bfloat162;
};
template <>
struct K_vec_<bfloat16, 2> {
  using Type = bf16_4_t;
};
template <>
struct K_vec_<bfloat16, 1> {
  using Type = bf16_8_t;
};
template <>
struct K_vec_bttn_<bfloat16> {
  using Type = bf16_8_t;
};
template <>
struct K_vec_I_bttn_<bfloat16, CacheType::NORMAL> {
  using Type = uint64_t;
};
template <>
struct K_vec_I_bttn_<bfloat16, CacheType::INT8> {
  using Type = uint64_t;
};
#endif  // ENABLE_BF16

//------------------------------------

template <typename T, int THREADS_PER_KEY>
struct K_vec_I_ {
  using Type = uint8_t;
};

#ifdef ENABLE_BF16
template <>
struct K_vec_I_<bfloat16, 4> {
  using Type = uint16_t;
};
template <>
struct K_vec_I_<bfloat16, 2> {
  using Type = uint32_t;
};
template <>
struct K_vec_I_<bfloat16, 1> {
  using Type = uint64_t;
};
#endif  // ENABLE_BF16

template <typename T, int V_VEC_SIZE>
struct V_vec_ {};
template <>
struct V_vec_<float, 1> {
  using Type = float;
};
template <>
struct V_vec_<float, 2> {
  using Type = float2;
};
template <>
struct V_vec_<float, 4> {
  using Type = float4;
};
template <>
struct V_vec_<float16, 2> {
  using Type = uint32_t;
};
template <>
struct V_vec_<float16, 4> {
  using Type = uint2;
};
template <>
struct V_vec_<float16, 8> {
  using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct V_vec_<bfloat16, 2> {
  using Type = __nv_bfloat162;
};
template <>
struct V_vec_<bfloat16, 4> {
  using Type = bf16_4_t;
};
template <>
struct V_vec_<bfloat16, 8> {
  using Type = bf16_8_t;
};
#endif  // ENABLE_BF16

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162 bf16hmul2(const __nv_bfloat162 x,
                                           const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl * fyl, fxh * fyh);
#else
  return __hmul2(x, y);
#endif
}

inline __device__ __nv_bfloat16 bf16hmul(const __nv_bfloat16 x,
                                         const __nv_bfloat16 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return __float2bfloat16(__bfloat162float(x) * __bfloat162float(y));
#else
  return __hmul(x, y);
#endif
}
#endif  // ENABLE_BF16

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifdef PADDLE_WITH_HIP
  asm volatile("v_cvt_f32_f16 %0, %1;\n" : "=v"(f) : "v"(h));
#else
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#endif
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
  uint16_t lo, hi;
#ifdef PADDLE_WITH_HIP
  lo = v & 0xffff;
  hi = (v >> 16) & 0xffff;
#else
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
#endif
  return make_float2(half_to_float(lo), half_to_float(hi));
}

inline __device__ uint32_t float2_to_half2(float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n"
               : "=r"(tmp.u32)
               : "f"(f.y), "f"(f.x));
#elif defined(PADDLE_WITH_HIP)
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u16[0]) : "v"(f.x));
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u16[1]) : "v"(f.y));
#else
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
  return tmp.u32;
}

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162 bf16hadd2(const __nv_bfloat162 x,
                                           const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
  return __hadd2(x, y);
#endif
}

inline __device__ float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float2 f_val;
  f_val.x = __low2float(val);
  f_val.y = __high2float(val);
  return f_val;
#else
  return __bfloat1622float2(val);
#endif
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val) {
  __nv_bfloat162 val_;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  val_ = __float22bfloat162_rn(val);
#else
  val_.x = __float2bfloat16_rn(val.x);
  val_.y = __float2bfloat16_rn(val.y);
#endif
  return val_;
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __nv_bfloat162 val2;
  val2.x = val;
  val2.y = val;
  return val2;
#else
  return __bfloat162bfloat162(val);
#endif
}

inline __device__ __nv_bfloat162 bf16hfma2(const __nv_bfloat162 x,
                                           const __nv_bfloat162 y,
                                           const __nv_bfloat162 z) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh, fzl, fzh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  fzl = __low2float(z);
  fzh = __high2float(z);
  return __floats2bfloat162_rn(fxl * fyl + fzl, fxh * fyh + fzh);
#else
  return __hfma2(x, y, z);
#endif
}
#endif  // ENABLE_BF16

inline __device__ float add(float a, float b) { return a + b; }

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
#ifdef PADDLE_WITH_HIP
  asm volatile("v_add_f16 %0, %1, %2;" : "=v"(c) : "v"(a), "v"(b));
#else
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#endif
  return c;
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
#ifdef PADDLE_WITH_HIP
  asm volatile("v_pk_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#else
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#endif
  return c;
}

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float2 add(uint32_t a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
  return a + b;
}

inline __device__ __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hadd2(a, b);
}

inline __device__ bf16_4_t add(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ bf16_8_t add(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float add(float a, __nv_bfloat16 b) {
  return a + __bfloat162float(b);
}

inline __device__ float2 add(__nv_bfloat162 a, float2 fb) {
  float2 fa = bf1622float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(bf16_4_t a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(bf16_8_t a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}
#endif  // ENABLE_BF16

template <typename T, typename IntT, CacheType CACHE_TYPE>
inline __device__ void mul_pointer_v2(T* c, float& a, IntT* b) {  // NOLINT
  printf("mul_pointer_v2 not support this case!\n");
}

template <typename T, typename FT, typename IntT, CacheType CACHE_TYPE>
inline __device__ void mul_pointer_v2(T* c, FT& a, IntT* b) {  // NOLINT
  printf("mul_pointer_v2 not support this case!\n");
}

template <typename T, typename FT, typename IntT, CacheType CACHE_TYPE>
inline __device__ void mul_pointer_v2(T* c, FT& a, FT& zp, IntT* b) {  // NOLINT
  printf("mul_pointer_v2 not support this case!\n");
}

template <>
inline __device__ void mul_pointer_v2<float4, float, uint8_t, CacheType::INT8>(
    float4* c, float& a, uint8_t* b) {  // NOLINT
  c->x = a * (static_cast<float>(b[0]) - 128.0);
  c->y = a * (static_cast<float>(b[1]) - 128.0);
  c->z = a * (static_cast<float>(b[2]) - 128.0);
  c->w = a * (static_cast<float>(b[3]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2<float4, float4, uint8_t, CacheType::INT8>(
    float4* c, float4& a, uint8_t* b) {  // NOLINT
  c->x = a.x * (static_cast<float>(b[0]) - 128.0);
  c->y = a.y * (static_cast<float>(b[1]) - 128.0);
  c->z = a.z * (static_cast<float>(b[2]) - 128.0);
  c->w = a.w * (static_cast<float>(b[3]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2<float4, float, uint32_t, CacheType::INT8>(
    float4* c, float& a, uint32_t* b) {  // NOLINT
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x = a * (static_cast<float>(b_tmp[0]) - 128.0);
  c->y = a * (static_cast<float>(b_tmp[1]) - 128.0);
  c->z = a * (static_cast<float>(b_tmp[2]) - 128.0);
  c->w = a * (static_cast<float>(b_tmp[3]) - 128.0);
}

template <>
inline __device__ void
mul_pointer_v2<float4, float4, uint32_t, CacheType::INT8>(float4* c,
                                                          float4& a,  // NOLINT
                                                          uint32_t* b) {
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x = a.x * (static_cast<float>(b_tmp[0]) - 128.0);
  c->y = a.y * (static_cast<float>(b_tmp[1]) - 128.0);
  c->z = a.z * (static_cast<float>(b_tmp[2]) - 128.0);
  c->w = a.w * (static_cast<float>(b_tmp[3]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2<float, float, uint8_t, CacheType::INT8>(
    float* c, float& a, float& zp, uint8_t* b) {  // NOLINT
  *c = a * (static_cast<float>(b[0]) - 128.0 - zp);
}

template <>
inline __device__ void
mul_pointer_v2<float2, float2, uint16_t, CacheType::INT8>(float2* c,
                                                          float2& a,   // NOLINT
                                                          float2& zp,  // NOLINT
                                                          uint16_t* b) {
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x = a.x * (static_cast<float>(b_tmp[0]) - 128.0 - zp.x);
  c->y = a.y * (static_cast<float>(b_tmp[1]) - 128.0 - zp.y);
}

template <>
inline __device__ void
mul_pointer_v2<float4, float4, uint32_t, CacheType::INT8>(float4* c,
                                                          float4& a,   // NOLINT
                                                          float4& zp,  // NOLINT
                                                          uint32_t* b) {
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x = a.x * (static_cast<float>(b_tmp[0]) - 128.0 - zp.x);
  c->y = a.y * (static_cast<float>(b_tmp[1]) - 128.0 - zp.y);
  c->z = a.z * (static_cast<float>(b_tmp[2]) - 128.0 - zp.z);
  c->w = a.w * (static_cast<float>(b_tmp[3]) - 128.0 - zp.w);
}

template <>
inline __device__ void
mul_pointer_v2<Float8_, Float8_, uint64_t, CacheType::INT8>(
    Float8_* c,
    Float8_& a,   // NOLINT
    Float8_& zp,  // NOLINT
    uint64_t* b) {
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x.x = a.x.x * (static_cast<float>(b_tmp[0]) - 128.0 - zp.x.x);
  c->x.y = a.x.y * (static_cast<float>(b_tmp[1]) - 128.0 - zp.x.y);
  c->y.x = a.y.x * (static_cast<float>(b_tmp[2]) - 128.0 - zp.y.x);
  c->y.y = a.y.y * (static_cast<float>(b_tmp[3]) - 128.0 - zp.y.y);
  c->z.x = a.z.x * (static_cast<float>(b_tmp[4]) - 128.0 - zp.z.x);
  c->z.y = a.z.y * (static_cast<float>(b_tmp[5]) - 128.0 - zp.z.y);
  c->w.x = a.w.x * (static_cast<float>(b_tmp[6]) - 128.0 - zp.w.x);
  c->w.y = a.w.y * (static_cast<float>(b_tmp[7]) - 128.0 - zp.w.y);
}

template <>
inline __device__ void mul_pointer_v2<float2, float, uint8_t, CacheType::INT8>(
    float2* c, float& a, uint8_t* b) {  // NOLINT
  c->x = a * (static_cast<float>(b[0]) - 128.0);
  c->y = a * (static_cast<float>(b[1]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2<float2, float2, uint8_t, CacheType::INT8>(
    float2* c, float2& a, uint8_t* b) {  // NOLINT
  c->x = a.x * (static_cast<float>(b[0]) - 128.0);
  c->y = a.y * (static_cast<float>(b[1]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2<float, float, uint8_t, CacheType::INT8>(
    float* c, float& a, uint8_t* b) {  // NOLINT
  c[0] = a * (static_cast<float>(b[0]) - 128.0);
}

template <>
inline __device__ void
mul_pointer_v2<uint32_t, float, uint8_t, CacheType::INT8>(uint32_t* c,
                                                          float& a,  // NOLINT
                                                          uint8_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(b[i]) - offset);
  }
}

template <>
inline __device__ void
mul_pointer_v2<uint32_t, float2, uint8_t, CacheType::INT8>(uint32_t* c,
                                                           float2& a,  // NOLINT
                                                           uint8_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);

  tmp_fp16[0] =
      static_cast<float16>(a.x) * (static_cast<float16>(b[0]) - offset);
  tmp_fp16[1] =
      static_cast<float16>(a.y) * (static_cast<float16>(b[1]) - offset);
}

template <>
inline __device__ void
mul_pointer_v2<uint32_t, float2, uint16_t, CacheType::INT8>(
    uint32_t* c,
    float2& a,   // NOLINT
    float2& zp,  // NOLINT
    uint16_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);

  tmp_fp16[0] =
      static_cast<float16>(a.x) *
      (static_cast<float16>(tmp_b[0]) - offset - static_cast<float16>(zp.x));
  tmp_fp16[1] =
      static_cast<float16>(a.y) *
      (static_cast<float16>(tmp_b[1]) - offset - static_cast<float16>(zp.y));
}

template <>
inline __device__ void mul_pointer_v2<uint2, float4, uint32_t, CacheType::INT8>(
    uint2* c, float4& a, float4& zp, uint32_t* b) {  // NOLINT
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);

  tmp_fp16[0] =
      static_cast<float16>(a.x) *
      (static_cast<float16>(tmp_b[0]) - offset - static_cast<float16>(zp.x));
  tmp_fp16[1] =
      static_cast<float16>(a.y) *
      (static_cast<float16>(tmp_b[1]) - offset - static_cast<float16>(zp.y));
  tmp_fp16[2] =
      static_cast<float16>(a.z) *
      (static_cast<float16>(tmp_b[2]) - offset - static_cast<float16>(zp.z));
  tmp_fp16[3] =
      static_cast<float16>(a.w) *
      (static_cast<float16>(tmp_b[3]) - offset - static_cast<float16>(zp.w));
}

template <>
inline __device__ void
mul_pointer_v2<uint4, Float8_, uint64_t, CacheType::INT8>(
    uint4* c,
    Float8_& a,   // NOLINT
    Float8_& zp,  // NOLINT
    uint64_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);

  tmp_fp16[0] =
      static_cast<float16>(a.x.x) *
      (static_cast<float16>(tmp_b[0]) - offset - static_cast<float16>(zp.x.x));
  tmp_fp16[1] =
      static_cast<float16>(a.x.y) *
      (static_cast<float16>(tmp_b[1]) - offset - static_cast<float16>(zp.x.y));
  tmp_fp16[2] =
      static_cast<float16>(a.y.x) *
      (static_cast<float16>(tmp_b[2]) - offset - static_cast<float16>(zp.y.x));
  tmp_fp16[3] =
      static_cast<float16>(a.y.y) *
      (static_cast<float16>(tmp_b[3]) - offset - static_cast<float16>(zp.y.y));
  tmp_fp16[4] =
      static_cast<float16>(a.z.x) *
      (static_cast<float16>(tmp_b[4]) - offset - static_cast<float16>(zp.z.x));
  tmp_fp16[5] =
      static_cast<float16>(a.z.y) *
      (static_cast<float16>(tmp_b[5]) - offset - static_cast<float16>(zp.z.y));
  tmp_fp16[6] =
      static_cast<float16>(a.w.x) *
      (static_cast<float16>(tmp_b[6]) - offset - static_cast<float16>(zp.w.x));
  tmp_fp16[7] =
      static_cast<float16>(a.w.y) *
      (static_cast<float16>(tmp_b[7]) - offset - static_cast<float16>(zp.w.y));
}

template <>
inline __device__ void mul_pointer_v2<uint4, uint4, uint64_t, CacheType::INT8>(
    uint4* c, uint4& a, uint4& zp, uint64_t* b) {  // NOLINT
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16* tmp_a = reinterpret_cast<float16*>(&a);
  float16* tmp_zp = reinterpret_cast<float16*>(&zp);
  float16 offset = static_cast<float16>(128.0);
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    tmp_fp16[i] =
        tmp_a[i] * (static_cast<float16>(tmp_b[i]) - offset - tmp_zp[i]);
  }
}

template <>
inline __device__ void mul_pointer_v2<uint2, float, uint8_t, CacheType::INT8>(
    uint2* c, float& a, uint8_t* b) {  // NOLINT
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(b[i]) - offset);
  }
}

template <>
inline __device__ void mul_pointer_v2<uint2, float4, uint8_t, CacheType::INT8>(
    uint2* c, float4& a, uint8_t* b) {  // NOLINT
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);

  tmp_fp16[0] =
      static_cast<float16>(a.x) * (static_cast<float16>(b[0]) - offset);
  tmp_fp16[1] =
      static_cast<float16>(a.y) * (static_cast<float16>(b[1]) - offset);
  tmp_fp16[2] =
      static_cast<float16>(a.z) * (static_cast<float16>(b[2]) - offset);
  tmp_fp16[3] =
      static_cast<float16>(a.w) * (static_cast<float16>(b[3]) - offset);
}

template <>
inline __device__ void mul_pointer_v2<uint4, float, uint8_t, CacheType::INT8>(
    uint4* c, float& a, uint8_t* b) {  // NOLINT
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(b[i]) - offset);
  }
}

template <>
inline __device__ void mul_pointer_v2<uint4, Float8_, uint8_t, CacheType::INT8>(
    uint4* c, Float8_& a, uint8_t* b) {  // NOLINT
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);

  tmp_fp16[0] =
      static_cast<float16>(a.x.x) * (static_cast<float16>(b[0]) - offset);
  tmp_fp16[1] =
      static_cast<float16>(a.x.y) * (static_cast<float16>(b[1]) - offset);
  tmp_fp16[2] =
      static_cast<float16>(a.y.x) * (static_cast<float16>(b[2]) - offset);
  tmp_fp16[3] =
      static_cast<float16>(a.y.y) * (static_cast<float16>(b[3]) - offset);
  tmp_fp16[4] =
      static_cast<float16>(a.z.x) * (static_cast<float16>(b[4]) - offset);
  tmp_fp16[5] =
      static_cast<float16>(a.z.y) * (static_cast<float16>(b[5]) - offset);
  tmp_fp16[6] =
      static_cast<float16>(a.w.x) * (static_cast<float16>(b[6]) - offset);
  tmp_fp16[7] =
      static_cast<float16>(a.w.y) * (static_cast<float16>(b[7]) - offset);
}

template <>
inline __device__ void mul_pointer_v2<uint4, float, uint64_t, CacheType::INT8>(
    uint4* c, float& a, uint64_t* b) {  // NOLINT
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(tmp_b[i]) - offset);
  }
}

template <>
inline __device__ void
mul_pointer_v2<uint4, Float8_, uint64_t, CacheType::INT8>(uint4* c,
                                                          Float8_& a,  // NOLINT
                                                          uint64_t* b) {
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 offset = static_cast<float16>(128.0);

  tmp_fp16[0] =
      static_cast<float16>(a.x.x) * (static_cast<float16>(tmp_b[0]) - offset);
  tmp_fp16[1] =
      static_cast<float16>(a.x.y) * (static_cast<float16>(tmp_b[1]) - offset);
  tmp_fp16[2] =
      static_cast<float16>(a.y.x) * (static_cast<float16>(tmp_b[2]) - offset);
  tmp_fp16[3] =
      static_cast<float16>(a.y.y) * (static_cast<float16>(tmp_b[3]) - offset);
  tmp_fp16[4] =
      static_cast<float16>(a.z.x) * (static_cast<float16>(tmp_b[4]) - offset);
  tmp_fp16[5] =
      static_cast<float16>(a.z.y) * (static_cast<float16>(tmp_b[5]) - offset);
  tmp_fp16[6] =
      static_cast<float16>(a.w.x) * (static_cast<float16>(tmp_b[6]) - offset);
  tmp_fp16[7] =
      static_cast<float16>(a.w.y) * (static_cast<float16>(tmp_b[7]) - offset);
}

template <>
inline __device__ void mul_pointer_v2<uint4, uint4, uint64_t, CacheType::INT8>(
    uint4* c, uint4& a, uint64_t* b) {  // NOLINT
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16* tmp_a = reinterpret_cast<float16*>(&a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    tmp_fp16[i] = tmp_a[i] * (static_cast<float16>(tmp_b[i]) - offset);
  }
}

inline __device__ static void convert_(__nv_bfloat16* result,
                                       uint32_t const& source) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[4];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);
  fp32_intermediates_casted[2] = __byte_perm(i8s, fp32_base, 0x7652);
  fp32_intermediates_casted[3] = __byte_perm(i8s, fp32_base, 0x7653);

#pragma unroll
  for (int ii = 0; ii < 4; ++ii) {
    fp32_intermediates[ii] -= (8388608.f + 128.f);
  }

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    bf16_result_ptr[ii] = __byte_perm(fp32_intermediates_casted[2 * ii + 0],
                                      fp32_intermediates_casted[2 * ii + 1],
                                      0x7632);
  }
#endif
}

template <>
inline __device__ void
mul_pointer_v2<__nv_bfloat162, float, uint8_t, CacheType::INT8>(
    __nv_bfloat162* c, float& a, uint8_t* b) {  // NOLINT
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(b)));
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    c_prime[i] *= a_prime;
  }
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<__nv_bfloat162, float2, uint8_t, CacheType::INT8>(
    __nv_bfloat162* c, float2& a, uint8_t* b) {  // NOLINT
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(b)));

  c_prime[0] *= static_cast<__nv_bfloat16>(a.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.y);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<__nv_bfloat162, float, uint16_t, CacheType::INT8>(
    __nv_bfloat162* c, float& a, uint16_t* b) {  // NOLINT
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  using Packed_Int8_t = typename packed_type<uint8_t, 2>::type;
  Packed_Int8_t int8_vec_4_val = *reinterpret_cast<Packed_Int8_t*>(b);
  uint8_t* int8_vec_pointer = reinterpret_cast<uint8_t*>(&int8_vec_4_val);

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(c);
  uint32_t const i8s = int8_vec_4_val;

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[2];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    fp32_intermediates[ii] -= (8388608.f + 128.f);
  }

  bf16_result_ptr[0] = __byte_perm(
      fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  __nv_bfloat16 scale = static_cast<__nv_bfloat16>(a);
  c->x *= scale;
  c->y *= scale;
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<__nv_bfloat162, float2, uint16_t, CacheType::INT8>(
    __nv_bfloat162* c, float2& a, uint16_t* b) {  // NOLINT
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  using Packed_Int8_t = typename packed_type<uint8_t, 2>::type;
  Packed_Int8_t int8_vec_4_val = *reinterpret_cast<Packed_Int8_t*>(b);
  uint8_t* int8_vec_pointer = reinterpret_cast<uint8_t*>(&int8_vec_4_val);

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(c);
  uint32_t const i8s = int8_vec_4_val;

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[2];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    fp32_intermediates[ii] -= (8388608.f + 128.f);
  }

  bf16_result_ptr[0] = __byte_perm(
      fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);
  c->x *= static_cast<__nv_bfloat16>(a.x);
  c->y *= static_cast<__nv_bfloat16>(a.y);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<__nv_bfloat162, float2, uint16_t, CacheType::INT8>(
    __nv_bfloat162* c, float2& a, float2& zp, uint16_t* b) {  // NOLINT
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  using Packed_Int8_t = typename packed_type<uint8_t, 2>::type;
  Packed_Int8_t int8_vec_4_val = *reinterpret_cast<Packed_Int8_t*>(b);
  uint8_t* int8_vec_pointer = reinterpret_cast<uint8_t*>(&int8_vec_4_val);

  uint32_t* bf16_result_ptr = reinterpret_cast<uint32_t*>(c);
  uint32_t const i8s = int8_vec_4_val;

  static constexpr uint32_t fp32_base = 0x4B000000;
  float fp32_intermediates[2];

  uint32_t* fp32_intermediates_casted =
      reinterpret_cast<uint32_t*>(fp32_intermediates);
  fp32_intermediates_casted[0] = __byte_perm(i8s, fp32_base, 0x7650);
  fp32_intermediates_casted[1] = __byte_perm(i8s, fp32_base, 0x7651);

#pragma unroll
  for (int ii = 0; ii < 2; ++ii) {
    fp32_intermediates[ii] -= (8388608.f + 128.f);
  }

  bf16_result_ptr[0] = __byte_perm(
      fp32_intermediates_casted[0], fp32_intermediates_casted[1], 0x7632);

  c->x -= static_cast<__nv_bfloat16>(zp.x);
  c->y -= static_cast<__nv_bfloat16>(zp.y);
  c->x *= static_cast<__nv_bfloat16>(a.x);
  c->y *= static_cast<__nv_bfloat16>(a.y);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_4_t, float, uint8_t, CacheType::INT8>(bf16_4_t* c,
                                                          float& a,  // NOLINT
                                                          uint8_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *reinterpret_cast<uint32_t*>(b));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c_prime[i] *= a_prime;
  }
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_4_t, float4, uint8_t, CacheType::INT8>(bf16_4_t* c,
                                                           float4& a,  // NOLINT
                                                           uint8_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *reinterpret_cast<uint32_t*>(b));

  c_prime[0] *= static_cast<__nv_bfloat16>(a.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.y);
  c_prime[2] *= static_cast<__nv_bfloat16>(a.z);
  c_prime[3] *= static_cast<__nv_bfloat16>(a.w);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_4_t, float4, uint32_t, CacheType::INT8>(
    bf16_4_t* c,
    float4& a,   // NOLINT
    float4& zp,  // NOLINT
    uint32_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *b);

  c_prime[0] -= static_cast<__nv_bfloat16>(zp.x);
  c_prime[1] -= static_cast<__nv_bfloat16>(zp.y);
  c_prime[2] -= static_cast<__nv_bfloat16>(zp.z);
  c_prime[3] -= static_cast<__nv_bfloat16>(zp.w);
  c_prime[0] *= static_cast<__nv_bfloat16>(a.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.y);
  c_prime[2] *= static_cast<__nv_bfloat16>(a.z);
  c_prime[3] *= static_cast<__nv_bfloat16>(a.w);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_4_t, float, uint32_t, CacheType::INT8>(bf16_4_t* c,
                                                           float& a,  // NOLINT
                                                           uint32_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c_prime[i] *= a_prime;
  }
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_4_t, float4, uint32_t, CacheType::INT8>(
    bf16_4_t* c,
    float4& a,  // NOLINT
    uint32_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *b);

  c_prime[0] *= static_cast<__nv_bfloat16>(a.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.y);
  c_prime[2] *= static_cast<__nv_bfloat16>(a.z);
  c_prime[3] *= static_cast<__nv_bfloat16>(a.w);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, float, uint8_t, CacheType::INT8>(bf16_8_t* c,
                                                          float& a,  // NOLINT
                                                          uint8_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  bf16_4_t* tmp_c = reinterpret_cast<bf16_4_t*>(c);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    mul_pointer_v2<bf16_4_t, float, uint8_t, CacheType::INT8>(
        tmp_c + i, a, b + 4 * i);
  }
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, Float8_, uint64_t, CacheType::INT8>(
    bf16_8_t* c,
    Float8_& a,   // NOLINT
    Float8_& zp,  // NOLINT
    uint64_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(b);
  convert_(c_prime, tmp_b[0]);
  convert_(c_prime + 4, tmp_b[1]);
  c_prime[0] -= static_cast<__nv_bfloat16>(zp.x.x);
  c_prime[1] -= static_cast<__nv_bfloat16>(zp.x.y);
  c_prime[2] -= static_cast<__nv_bfloat16>(zp.y.x);
  c_prime[3] -= static_cast<__nv_bfloat16>(zp.y.y);
  c_prime[4] -= static_cast<__nv_bfloat16>(zp.z.x);
  c_prime[5] -= static_cast<__nv_bfloat16>(zp.z.y);
  c_prime[6] -= static_cast<__nv_bfloat16>(zp.w.x);
  c_prime[7] -= static_cast<__nv_bfloat16>(zp.w.y);
  c_prime[0] *= static_cast<__nv_bfloat16>(a.x.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.x.y);
  c_prime[2] *= static_cast<__nv_bfloat16>(a.y.x);
  c_prime[3] *= static_cast<__nv_bfloat16>(a.y.y);
  c_prime[4] *= static_cast<__nv_bfloat16>(a.z.x);
  c_prime[5] *= static_cast<__nv_bfloat16>(a.z.y);
  c_prime[6] *= static_cast<__nv_bfloat16>(a.w.x);
  c_prime[7] *= static_cast<__nv_bfloat16>(a.w.y);
#ifdef DEBUG_BLHA
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0) {
    printf("mul_pointer_v2 float8 bf16_8 int8\n");
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.x.x,
           zp.x.x,
           *b,
           static_cast<float>(c->x.x));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.x.y,
           zp.x.y,
           *b,
           static_cast<float>(c->x.y));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.y.x,
           zp.y.x,
           *b,
           static_cast<float>(c->y.x));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.y.y,
           zp.y.y,
           *b,
           static_cast<float>(c->y.y));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.z.x,
           zp.z.x,
           *b,
           static_cast<float>(c->z.x));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.z.y,
           zp.z.y,
           *b,
           static_cast<float>(c->z.y));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.w.x,
           zp.w.x,
           *b,
           static_cast<float>(c->w.x));
    printf("scale: %f, zp: %f, b: %lu\n, c: %f\n",
           a.w.y,
           zp.w.y,
           *b,
           static_cast<float>(c->w.y));
  }
#endif
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, bf16_8_t, uint64_t, CacheType::INT8>(
    bf16_8_t* c,
    bf16_8_t& a,   // NOLINT
    bf16_8_t& zp,  // NOLINT
    uint64_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(b);
  convert_(c_prime, tmp_b[0]);
  convert_(c_prime + 4, tmp_b[1]);
  c_prime[0] -= zp.x.x;
  c_prime[1] -= zp.x.y;
  c_prime[2] -= zp.y.x;
  c_prime[3] -= zp.y.y;
  c_prime[4] -= zp.z.x;
  c_prime[5] -= zp.z.y;
  c_prime[6] -= zp.w.x;
  c_prime[7] -= zp.w.y;
  c_prime[0] *= a.x.x;
  c_prime[1] *= a.x.y;
  c_prime[2] *= a.y.x;
  c_prime[3] *= a.y.y;
  c_prime[4] *= a.z.x;
  c_prime[5] *= a.z.y;
  c_prime[6] *= a.w.x;
  c_prime[7] *= a.w.y;
#ifdef DEBUG_BLHA
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0) {
    printf("mul_pointer_v2 float8 bf16_8 int8\n");
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.x.x),
           static_cast<float>(zp.x.x),
           *b,
           static_cast<float>(c->x.x));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.x.y),
           static_cast<float>(zp.x.y),
           *b,
           static_cast<float>(c->x.y));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.y.x),
           static_cast<float>(zp.y.x),
           *b,
           static_cast<float>(c->y.x));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.y.y),
           static_cast<float>(zp.y.y),
           *b,
           static_cast<float>(c->y.y));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.z.x),
           static_cast<float>(zp.z.x),
           *b,
           static_cast<float>(c->z.x));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.z.y),
           static_cast<float>(zp.z.y),
           *b,
           static_cast<float>(c->z.y));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.w.x),
           static_cast<float>(zp.w.x),
           *b,
           static_cast<float>(c->w.x));
    printf("scale: %f, zp: %f, b: %u\n, c: %f\n",
           static_cast<float>(a.w.y),
           static_cast<float>(zp.w.y),
           *b,
           static_cast<float>(c->w.y));
  }
#endif
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, Float8_, uint8_t, CacheType::INT8>(
    bf16_8_t* c,
    Float8_& a,  // NOLINT
    uint8_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(b);
  convert_(c_prime, tmp_b[0]);
  convert_(c_prime + 4, tmp_b[1]);
  c_prime[0] *= static_cast<__nv_bfloat16>(a.x.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.x.y);
  c_prime[2] *= static_cast<__nv_bfloat16>(a.y.x);
  c_prime[3] *= static_cast<__nv_bfloat16>(a.y.y);
  c_prime[4] *= static_cast<__nv_bfloat16>(a.z.x);
  c_prime[5] *= static_cast<__nv_bfloat16>(a.z.y);
  c_prime[6] *= static_cast<__nv_bfloat16>(a.w.x);
  c_prime[7] *= static_cast<__nv_bfloat16>(a.w.y);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, float, uint64_t, CacheType::INT8>(bf16_8_t* c,
                                                           float& a,  // NOLINT
                                                           uint64_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  bf16_4_t* tmp_c = reinterpret_cast<bf16_4_t*>(c);
  uint64_t bb = *b;
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(&bb);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    mul_pointer_v2<bf16_4_t, float, uint32_t, CacheType::INT8>(
        tmp_c + i, a, tmp_b + i);
  }
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, Float8_, uint64_t, CacheType::INT8>(
    bf16_8_t* c,
    Float8_& a,  // NOLINT
    uint64_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(b);
  convert_(c_prime, tmp_b[0]);
  convert_(c_prime + 4, tmp_b[1]);
  c_prime[0] *= static_cast<__nv_bfloat16>(a.x.x);
  c_prime[1] *= static_cast<__nv_bfloat16>(a.x.y);
  c_prime[2] *= static_cast<__nv_bfloat16>(a.y.x);
  c_prime[3] *= static_cast<__nv_bfloat16>(a.y.y);
  c_prime[4] *= static_cast<__nv_bfloat16>(a.z.x);
  c_prime[5] *= static_cast<__nv_bfloat16>(a.z.y);
  c_prime[6] *= static_cast<__nv_bfloat16>(a.w.x);
  c_prime[7] *= static_cast<__nv_bfloat16>(a.w.y);
#else
  assert(false);
#endif
}

template <>
inline __device__ void
mul_pointer_v2<bf16_8_t, bf16_8_t, uint64_t, CacheType::INT8>(
    bf16_8_t* c,
    bf16_8_t& a,  // NOLINT
    uint64_t* b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(b);
  convert_(c_prime, tmp_b[0]);
  convert_(c_prime + 4, tmp_b[1]);
  c_prime[0] *= a.x.x;
  c_prime[1] *= a.x.y;
  c_prime[2] *= a.y.x;
  c_prime[3] *= a.y.y;
  c_prime[4] *= a.z.x;
  c_prime[5] *= a.z.y;
  c_prime[6] *= a.w.x;
  c_prime[7] *= a.w.y;
#ifdef DEBUG_BLHA
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0) {
    printf("mul_pointer_v2 float8 bf16_8 int8\n");
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.x.x),
           *b,
           static_cast<float>(c->x.x));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.x.y),
           *b,
           static_cast<float>(c->x.y));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.y.x),
           *b,
           static_cast<float>(c->y.x));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.y.y),
           *b,
           static_cast<float>(c->y.y));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.z.x),
           *b,
           static_cast<float>(c->z.x));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.z.y),
           *b,
           static_cast<float>(c->z.y));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.w.x),
           *b,
           static_cast<float>(c->w.x));
    printf("scale: %f, b: %lu\n, c: %f\n",
           static_cast<float>(a.w.y),
           *b,
           static_cast<float>(c->w.y));
  }
#endif
#else
  assert(false);
#endif
}

template <typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template <>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

#ifdef ENABLE_BF16
template <>
inline __device__ __nv_bfloat162 mul(float a, __nv_bfloat162 b) {
  __nv_bfloat162 ret;
  ret.x = static_cast<__nv_bfloat16>(a) * b.x;
  ret.y = static_cast<__nv_bfloat16>(a) * b.y;
  return ret;
}

template <>
inline __device__ bf16_4_t mul(float a, bf16_4_t b) {
  bf16_4_t ret;
  ret.x = mul<__nv_bfloat162, float, __nv_bfloat162>(a, b.x);
  ret.y = mul<__nv_bfloat162, float, __nv_bfloat162>(a, b.y);
  return ret;
}

template <>
inline __device__ bf16_8_t mul(float a, bf16_8_t b) {
  bf16_8_t ret;
  ret.x = mul<__nv_bfloat162, float, __nv_bfloat162>(a, b.x);
  ret.y = mul<__nv_bfloat162, float, __nv_bfloat162>(a, b.y);
  ret.z = mul<__nv_bfloat162, float, __nv_bfloat162>(a, b.z);
  ret.w = mul<__nv_bfloat162, float, __nv_bfloat162>(a, b.w);
  return ret;
}
#endif  // ENABLE_BF16

template <>
inline __device__ uint32_t mul(float a, uint32_t b) {
  union {
    float16 out[2];
    uint32_t t_out;
  };

  union {
    float16 in[2];
    uint32_t t_in;
  };
  t_in = b;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    out[i] = static_cast<float16>(a) * in[i];
  }
  return t_out;
}

template <>
inline __device__ float16 mul(float a, float16 b) {
  return static_cast<float16>(a) * b;
}

template <>
inline __device__ uint2 mul(float a, uint2 b) {
  union {
    uint2 tmp_in;
    float16 tmp_in_fp16[4];
  };
  tmp_in = b;
  union {
    uint2 ret;
    float16 tmp_out_fp16[4];
  };

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    tmp_out_fp16[i] = mul<float16, float, float16>(a, tmp_in_fp16[i]);
  }
  return ret;
}

template <>
inline __device__ uint4 mul(float a, uint4 b) {
  union {
    uint4 tmp_in;
    float16 tmp_in_fp16[8];
  };
  tmp_in = b;
  union {
    uint4 ret;
    float16 tmp_out_fp16[8];
  };
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp_out_fp16[i] = mul<float16, float, float16>(a, tmp_in_fp16[i]);
  }
  return ret;
}

template <>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

template <>
inline __device__ Float8_ mul(float a, Float8_ b) {
  Float8_ c;
  c.x = mul<float2, float, float2>(a, b.x);
  c.y = mul<float2, float, float2>(a, b.y);
  c.z = mul<float2, float, float2>(a, b.z);
  c.w = mul<float2, float, float2>(a, b.w);
  return c;
}

template <>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template <>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template <>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
#ifdef PADDLE_WITH_HIP
  asm volatile("v_mul_f16 %0, %1, %2;" : "=v"(c) : "v"(a), "v"(b));
#else
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#endif
  return c;
}

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
#ifdef PADDLE_WITH_HIP
  asm volatile("v_pk_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#else
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#endif
  return c;
}

template <>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template <>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

template <>
inline __device__ uint32_t mul(uint32_t a, float b) {
  float2 tmp = half2_to_float2(a);
  float2 tmp_res;
  tmp_res.x = tmp.x * b;
  tmp_res.y = tmp.y * b;
  uint32_t res = float2_to_half2(tmp_res);
  return res;
}

template <>
inline __device__ uint32_t mul(uint32_t a, float2 b) {
  float2 tmp = half2_to_float2(a);
  float2 tmp_res;
  tmp_res.x = tmp.x * b.x;
  tmp_res.y = tmp.y * b.y;
  uint32_t res = float2_to_half2(tmp_res);
  return res;
}

template <>
inline __device__ float2 mul(uint32_t a, float b) {
  float2 tmp = half2_to_float2(a);
  float2 res;
  res.x = tmp.x * b;
  res.y = tmp.y * b;
  return res;
}

template <>
inline __device__ uint2 mul(uint2 a, float b) {
  uint2 res;
  res.x = mul<uint32_t, uint32_t, float>(a.x, b);
  res.y = mul<uint32_t, uint32_t, float>(a.y, b);
  return res;
}

template <>
inline __device__ uint2 mul(uint2 a, float4 b) {
  Float4_& b_ = *reinterpret_cast<Float4_*>(&b);
  uint2 res;
  res.x = mul<uint32_t, uint32_t, float2>(a.x, b_.x);
  res.y = mul<uint32_t, uint32_t, float2>(a.y, b_.y);
  return res;
}

template <>
inline __device__ uint4 mul(uint4 a, float b) {
  uint4 res;
  res.x = mul<uint32_t, uint32_t, float>(a.x, b);
  res.y = mul<uint32_t, uint32_t, float>(a.y, b);
  res.z = mul<uint32_t, uint32_t, float>(a.z, b);
  res.w = mul<uint32_t, uint32_t, float>(a.w, b);
  return res;
}

template <>
inline __device__ uint4 mul(uint4 a, Float8_ b) {
  uint4 res;
  res.x = mul<uint32_t, uint32_t, float2>(a.x, b.x);
  res.y = mul<uint32_t, uint32_t, float2>(a.y, b.y);
  res.z = mul<uint32_t, uint32_t, float2>(a.z, b.z);
  res.w = mul<uint32_t, uint32_t, float2>(a.w, b.w);
  return res;
}

template <>
inline __device__ float2 mul(float2 a, float b) {
  float2 res;
  res.x = a.x * b;
  res.y = a.y * b;
  return res;
}

template <>
inline __device__ float2 mul(float2 a, uint32_t b) {
  float2 tmp_b = half2_to_float2(b);
  float2 res;
  res.x = a.x * tmp_b.x;
  res.y = a.y * tmp_b.y;
  return res;
}

template <>
inline __device__ float4 mul(float4 a, float b) {
  float4 res;
  res.x = a.x * b;
  res.y = a.y * b;
  res.z = a.z * b;
  res.w = a.w * b;
  return res;
}

#ifdef ENABLE_BF16
template <>
inline __device__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __hmul(a, b);
#else
  return bf16hmul(a, b);
#endif
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  return bf16hmul2(a, b);
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
  return mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template <>
inline __device__ bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return c;
}

template <>
inline __device__ bf16_4_t mul(__nv_bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return c;
}

template <>
inline __device__ bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return c;
}

template <>
inline __device__ bf16_8_t mul(__nv_bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.w);
  return c;
}

template <>
inline __device__ float mul(__nv_bfloat16 a, __nv_bfloat16 b) {
  float fa = static_cast<float>(a);
  float fb = static_cast<float>(b);
  return fa * fb;
}

template <>
inline __device__ float mul(__nv_bfloat16 a, float b) {
  return __bfloat162float(a) * b;
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, float b) {
  __nv_bfloat162 res;
  __nv_bfloat162 _bf16 = __float2bfloat162_rn(b);
  res = bf16hmul2(a, _bf16);
  return res;
}

template <>
inline __device__ __nv_bfloat162 mul(float2 a, float2 b) {
  float2 res = mul<float2, float2, float2>(a, b);
  __nv_bfloat162 bf16_res = float22bf162(res);
  return bf16_res;
}

template <>
inline __device__ __nv_bfloat162 mul(__nv_bfloat162 a, float2 b) {
  float2 a_ = bf1622float2(a);
  float2 res = mul<float2, float2, float2>(a_, b);
  __nv_bfloat162 bf16_res = float22bf162(res);
  return bf16_res;
}

template <>
inline __device__ bf16_4_t mul(bf16_4_t a, float b) {
  __nv_bfloat162 s = __float2bfloat162_rn(b);
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, s);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, s);
  return c;
}

template <>
inline __device__ bf16_4_t mul(bf16_4_t a, float4 b) {
  Float4_& b_ = *reinterpret_cast<Float4_*>(&b);
  float2 a1 = bf1622float2(a.x);
  float2 a2 = bf1622float2(a.y);

  bf16_4_t c;
  c.x = mul<__nv_bfloat162, float2, float2>(a1, b_.x);
  c.y = mul<__nv_bfloat162, float2, float2>(a2, b_.y);
  return c;
}

template <>
inline __device__ bf16_8_t mul(bf16_8_t a, float b) {
  __nv_bfloat162 s = __float2bfloat162_rn(b);
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, s);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, s);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z, s);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w, s);
  return c;
}

template <>
inline __device__ bf16_8_t mul(bf16_8_t a, Float8_ b) {
  float2 a1 = bf1622float2(a.x);
  float2 a2 = bf1622float2(a.y);
  float2 a3 = bf1622float2(a.z);
  float2 a4 = bf1622float2(a.w);

  bf16_8_t c;
  c.x = mul<__nv_bfloat162, float2, float2>(a1, b.x);
  c.y = mul<__nv_bfloat162, float2, float2>(a2, b.y);
  c.z = mul<__nv_bfloat162, float2, float2>(a3, b.z);
  c.w = mul<__nv_bfloat162, float2, float2>(a4, b.w);
  return c;
}

template <>
inline __device__ float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template <>
inline __device__ float2 mul(__nv_bfloat16 a, __nv_bfloat162 b) {
  return mul<float2, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template <>
inline __device__ Float4_ mul(bf16_4_t a, bf16_4_t b) {
  Float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return fc;
}

template <>
inline __device__ Float4_ mul(__nv_bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return fc;
}

template <>
inline __device__ Float8_ mul(bf16_8_t a, bf16_8_t b) {
  Float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return fc;
}

template <>
inline __device__ Float8_ mul(__nv_bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fc;
  fc.x = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  fc.z = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  fc.w = mul<float2, __nv_bfloat162, __nv_bfloat162>(s, b.w);
  return fc;
}
#endif  // ENABLE_BF16

template <typename Qk_vec, typename Qk_vec_RoPE>
inline __device__ Qk_vec apply_rotary_emb(Qk_vec input_left,
                                          Qk_vec input_right,
                                          Qk_vec_RoPE cos_emb,
                                          Qk_vec_RoPE sin_emb,
                                          float alpha) {
  Qk_vec res1 = mul<Qk_vec, Qk_vec, Qk_vec_RoPE>(input_left, cos_emb);
  Qk_vec res2 = mul<Qk_vec, Qk_vec, Qk_vec_RoPE>(input_right, sin_emb);
  res2 = mul<Qk_vec, Qk_vec, float>(res2, alpha);
  return add(res1, res2);
}

inline __device__ float sum(float v) { return v; }
inline __device__ float sum(float2 v) { return v.x + v.y; }
inline __device__ float sum(float4 v) { return v.x + v.y + v.z + v.w; }
inline __device__ float sum(uint16_t v) { return half_to_float(v); }
inline __device__ float sum(uint32_t v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

inline __device__ float sum(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(c);
}

inline __device__ float sum(uint4 v) {
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}

#ifdef ENABLE_BF16
inline __device__ float sum(__nv_bfloat162 v) {
  float2 vf = bf1622float2(v);
  return vf.x + vf.y;
}

inline __device__ float sum(bf16_4_t v) { return sum(v.x) + sum(v.y); }

inline __device__ float sum(bf16_8_t v) {
  return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
}
#endif  // ENABLE_BF16

template <typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template <typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

inline __device__ constexpr uint32_t shfl_mask(int threads) {
  return threads == 32 ? uint32_t(-1) : (1u << threads) - 1u;
}

template <typename T>
inline __device__ __host__ T div_up(T m, T n) {
  return (m + n - 1) / n;
}

inline __device__ float fma(float a, float b, float c) { return a * b + c; }

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ float2 fma(float2 a, uint32_t b, float2 c) {
  float2 tmp_b = half2_to_float2(b);
  float2 d = fma(a, tmp_b, c);
  return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
#ifdef PADDLE_WITH_HIP
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3;\n"
               : "=v"(d)
               : "v"(a), "v"(b), "v"(c));
#else
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
#endif
  return d;
}

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162 fma(float a, float2 b, __nv_bfloat162 c) {
  return bf16hfma2(__float2bfloat162_rn(a), float22bf162(b), c);
}

inline __device__ bf16_4_t fma(float a, Float4_ b, bf16_4_t c) {
  bf16_4_t d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}
#endif  // ENABLE_BF16

inline __device__ uint32_t h0_h0(uint16_t a) {
  uint32_t b;
#ifdef PADDLE_WITH_HIP
  b = (a << 16) | a;
#else
  asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
#endif
  return b;
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

#ifdef ENABLE_BF16

inline __device__ __nv_bfloat162 fma(__nv_bfloat162 a,
                                     __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
  return bf16hfma2(a, b, c);
}

inline __device__ __nv_bfloat162 fma(__nv_bfloat16 a,
                                     __nv_bfloat162 b,
                                     __nv_bfloat162 c) {
  return bf16hfma2(bf162bf162(a), b, c);
}

inline __device__ bf16_4_t fma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
  bf16_4_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ bf16_4_t fma(__nv_bfloat16 a, bf16_4_t b, bf16_4_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ bf16_8_t fma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
  bf16_8_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ bf16_8_t fma(__nv_bfloat16 a, bf16_8_t b, bf16_8_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(__nv_bfloat16 a, __nv_bfloat16 b, float fc) {
  return __bfloat162float(a) * __bfloat162float(b) + fc;
}

inline __device__ float2 fma(__nv_bfloat162 a, __nv_bfloat162 b, float2 fc) {
  float2 fa = bf1622float2(a);
  float2 fb = bf1622float2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(__nv_bfloat16 a, __nv_bfloat162 b, float2 fc) {
  return fma(bf162bf162(a), b, fc);
}

inline __device__ Float4_ fma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(__nv_bfloat16 a, bf16_4_t b, Float4_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(__nv_bfloat16 a, bf16_8_t b, Float8_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}
#endif  // ENABLE_BF16

inline __device__ float cast_to_float(float u) { return u; }

inline __device__ float2 cast_to_float(float2 u) { return u; }

inline __device__ float4 cast_to_float(float4 u) { return u; }

inline __device__ float2 cast_to_float(uint32_t u) {
  return half2_to_float2(u);
}

inline __device__ Float4_ cast_to_float(uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

inline __device__ Float8_ cast_to_float(uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}

inline __device__ Float4_ cast_to_float(Float4_ u) { return u; }

inline __device__ Float8_ cast_to_float(Float8_ u) { return u; }

#ifdef ENABLE_BF16
inline __device__ float cast_to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}

inline __device__ float2 cast_to_float(__nv_bfloat162 u) {
  return bf1622float2(u);
}

inline __device__ Float4_ cast_to_float(bf16_4_t u) {
  Float4_ tmp;
  tmp.x = bf1622float2(u.x);
  tmp.y = bf1622float2(u.y);
  return tmp;
}

inline __device__ Float8_ cast_to_float(bf16_8_t u) {
  Float8_ tmp;
  tmp.x = bf1622float2(u.x);
  tmp.y = bf1622float2(u.y);
  tmp.z = bf1622float2(u.z);
  tmp.w = bf1622float2(u.w);
  return tmp;
}
#endif  // ENABLE_BF16

template <typename T>
inline __device__ T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}

template <typename T, typename D, CacheType CACHE_TYPE>
inline __device__ T round_tmp(D val) {
  printf("round_tmp not support this case!\n");
}
template <typename T, typename D, CacheType CACHE_TYPE>
inline __device__ T round_tmp(D val1, D val2) {
  printf("round_tmp not support this case!\n");
}

template <>
inline __device__ uint8_t
round_tmp<uint8_t, float, CacheType::INT8>(float val) {
  float quant_value = roundWithTiesToEven(val);
  quant_value = quant_value > 127.0f ? 127.0f : quant_value;
  quant_value = quant_value < -127.0f ? -127.0f : quant_value;
  return static_cast<uint8_t>(quant_value + 128.0);
}

template <>
inline __device__ uint8_t
round_tmp<uint8_t, float16, CacheType::INT8>(float16 val) {
  float quant_value = roundWithTiesToEven(static_cast<float>(val));
  quant_value = quant_value > 127.0f ? 127.0f : quant_value;
  quant_value = quant_value < -127.0f ? -127.0f : quant_value;
  return static_cast<uint8_t>(quant_value + 128.0);
}

#ifndef PADDLE_WITH_HIP
template <>
inline __device__ uint8_t
round_tmp<uint8_t, __nv_bfloat16, CacheType::INT8>(__nv_bfloat16 val) {
  float quant_value =
      static_cast<float>(roundWithTiesToEven(static_cast<float>(val)));
  quant_value = quant_value > 127.0f ? 127.0f : quant_value;
  quant_value = quant_value < -127.0f ? -127.0f : quant_value;
  return static_cast<uint8_t>(quant_value + 128.0);
}
#endif

template <>
inline __device__ uint16_t
round_tmp<uint16_t, float2, CacheType::INT8>(float2 val) {
  union {
    uint16_t ret;
    uint8_t tmp[2];
  };
  tmp[0] = round_tmp<uint8_t, float, CacheType::INT8>(val.x);
  tmp[1] = round_tmp<uint8_t, float, CacheType::INT8>(val.y);
  return ret;
}

template <>
inline __device__ uint32_t
round_tmp<uint32_t, float4, CacheType::INT8>(float4 val) {
  union {
    uint32_t ret;
    uint8_t tmp[4];
  };
  tmp[0] = round_tmp<uint8_t, float, CacheType::INT8>(val.x);
  tmp[1] = round_tmp<uint8_t, float, CacheType::INT8>(val.y);
  tmp[2] = round_tmp<uint8_t, float, CacheType::INT8>(val.z);
  tmp[3] = round_tmp<uint8_t, float, CacheType::INT8>(val.w);
  return ret;
}

template <>
inline __device__ uint16_t
round_tmp<uint16_t, uint32_t, CacheType::INT8>(uint32_t val) {
  union {
    uint8_t int8[2];
    uint16_t ret;
  };
  union {
    float16 fp16[2];
    uint32_t tmp;
  };
  tmp = val;

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int8[i] = round_tmp<uint8_t, float16, CacheType::INT8>(fp16[i]);
  }

  return ret;
}

template <>
inline __device__ uint32_t
round_tmp<uint32_t, uint2, CacheType::INT8>(uint2 val) {
  union {
    uint8_t int8[4];
    uint32_t ret;
  };

  union {
    uint2 ui2;
    float16 tmp_fp16[4];
  };
  ui2 = val;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int8[i] = round_tmp<uint8_t, float16, CacheType::INT8>(tmp_fp16[i]);
  }
  return ret;
}

template <>
inline __device__ uint64_t
round_tmp<uint64_t, uint4, CacheType::INT8>(uint4 val) {
  union {
    uint8_t int8[8];
    uint64_t ret;
  };

  union {
    uint4 ui4;
    float16 tmp_fp16[8];
  };
  ui4 = val;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int8[i] = round_tmp<uint8_t, float16, CacheType::INT8>(tmp_fp16[i]);
  }
  return ret;
}

template <>
inline __device__ uint16_t
round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(__nv_bfloat162 val) {
  union {
    uint8_t tmp[2];
    uint16_t ret;
  };
  tmp[0] = round_tmp<uint8_t, __nv_bfloat16, CacheType::INT8>(val.x);
  tmp[1] = round_tmp<uint8_t, __nv_bfloat16, CacheType::INT8>(val.y);
  return ret;
}

template <>
inline __device__ uint32_t
round_tmp<uint32_t, bf16_4_t, CacheType::INT8>(bf16_4_t val) {
  union {
    uint16_t tmp[2];
    uint32_t ret;
  };
  tmp[0] = round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(val.x);
  tmp[1] = round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(val.y);
  return ret;
}

template <>
inline __device__ uint64_t
round_tmp<uint64_t, bf16_8_t, CacheType::INT8>(bf16_8_t val) {
  union {
    uint16_t int16[4];
    uint64_t int64;
  };
  int16[0] = round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(val.x);
  int16[1] = round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(val.y);
  int16[2] = round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(val.z);
  int16[3] = round_tmp<uint16_t, __nv_bfloat162, CacheType::INT8>(val.w);
  return int64;
}

inline __device__ float2 rotary_embedding_coefficient(const int zid,
                                                      const int rot_embed_dim,
                                                      const float t_step,
                                                      const float rope_theta) {
  const float inv_freq =
      t_step / pow(rope_theta,
                   static_cast<float>(zid) / static_cast<float>(rot_embed_dim));
  return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v,
                                                    const float2 coef) {
  float2 rot_v;
  rot_v.x = coef.x * v.x - coef.y * v.y;
  rot_v.y = coef.x * v.y + coef.y * v.x;
  return rot_v;
}

inline __device__ float2 rotary_embedding_transform(const float2 v,
                                                    const float2 cos,
                                                    const float2 sin) {
  float2 rot_v;
  rot_v.x = v.x * cos.x - v.y * sin.x;
  rot_v.y = v.y * cos.y + v.x * sin.y;
  return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v,
                                                      const float2 coef) {
  float2 fv = half2_to_float2(v);
  float2 rot_fv = rotary_embedding_transform(fv, coef);
  return float2_to_half2(rot_fv);
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v,
                                                      const uint32_t cos,
                                                      const uint32_t sin) {
  float2 fv = half2_to_float2(v);
  float2 fcos = half2_to_float2(cos);
  float2 fsin = half2_to_float2(sin);
  float2 rot_fv = rotary_embedding_transform(fv, fcos, fsin);
  return float2_to_half2(rot_fv);
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v,
                                                      const float2 cos,
                                                      const float2 sin) {
  float2 fv = half2_to_float2(v);
  float2 rot_fv = rotary_embedding_transform(fv, cos, sin);
  return float2_to_half2(rot_fv);
}

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162
rotary_embedding_transform(const __nv_bfloat162 v, const float2 coef) {
  float2 fv = bf1622float2(v);
  float2 rot_fv = rotary_embedding_transform(fv, coef);
  return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}

inline __device__ __nv_bfloat162
rotary_embedding_transform(const __nv_bfloat162 v,
                           const __nv_bfloat162 cos,
                           const __nv_bfloat162 sin) {
  float2 fv = bf1622float2(v);
  float2 fcos = bf1622float2(cos);
  float2 fsin = bf1622float2(sin);
  float2 rot_fv = rotary_embedding_transform(fv, fcos, fsin);
  return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}

inline __device__ __nv_bfloat162 rotary_embedding_transform(
    const __nv_bfloat162 v, const float2 cos, const float2 sin) {
  float2 fv = bf1622float2(v);
  float2 rot_fv = rotary_embedding_transform(fv, cos, sin);
  return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}

#endif

inline __device__ void apply_rotary_embedding(float& q,      // NOLINT
                                              float& k,      // NOLINT
                                              float& cos,    // NOLINT
                                              float& sin) {  // NOLINT
  return;
}

inline __device__ void apply_rotary_embedding(float2& q,      // NOLINT
                                              float2& k,      // NOLINT
                                              float2& cos,    // NOLINT
                                              float2& sin) {  // NOLINT
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(float4& q,      // NOLINT
                                              float4& k,      // NOLINT
                                              float4& cos,    // NOLINT
                                              float4& sin) {  // NOLINT
  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  q_.x = rotary_embedding_transform(q_.x, cos_.x, sin_.x);
  k_.x = rotary_embedding_transform(k_.x, cos_.x, sin_.x);
  q_.y = rotary_embedding_transform(q_.y, cos_.y, sin_.y);
  k_.y = rotary_embedding_transform(k_.y, cos_.y, sin_.y);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,      // NOLINT
                                              uint32_t& k,      // NOLINT
                                              uint32_t& cos,    // NOLINT
                                              uint32_t& sin) {  // NOLINT
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,    // NOLINT
                                              uint32_t& k,    // NOLINT
                                              float2& cos,    // NOLINT
                                              float2& sin) {  // NOLINT
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(uint2& q,      // NOLINT
                                              uint2& k,      // NOLINT
                                              uint2& cos,    // NOLINT
                                              uint2& sin) {  // NOLINT
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
}

inline __device__ void apply_rotary_embedding(uint2& q,       // NOLINT
                                              uint2& k,       // NOLINT
                                              float4& cos,    // NOLINT
                                              float4& sin) {  // NOLINT
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  q.x = rotary_embedding_transform(q.x, cos_.x, sin_.x);
  k.x = rotary_embedding_transform(k.x, cos_.x, sin_.x);
  q.y = rotary_embedding_transform(q.y, cos_.y, sin_.y);
  k.y = rotary_embedding_transform(k.y, cos_.y, sin_.x);
}

inline __device__ void apply_rotary_embedding(uint4& q,      // NOLINT
                                              uint4& k,      // NOLINT
                                              uint4& cos,    // NOLINT
                                              uint4& sin) {  // NOLINT
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(uint4& q,        // NOLINT
                                              uint4& k,        // NOLINT
                                              Float8_& cos,    // NOLINT
                                              Float8_& sin) {  // NOLINT
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(float& q,  // NOLINT
                                              int zid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  return;
}

inline __device__ void apply_rotary_embedding(float& q,  // NOLINT
                                              float& k,  // NOLINT
                                              int zid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  return;
}

inline __device__ void apply_rotary_embedding(float2& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef = rotary_embedding_coefficient(
      2 * tid, rot_embed_dim, float_t_step, rope_theta);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float2& q,  // NOLINT
                                              float2& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef = rotary_embedding_coefficient(
      2 * tid, rot_embed_dim, float_t_step, rope_theta);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  const auto coef0 = rotary_embedding_coefficient(
      4 * tid, rot_embed_dim, float_t_step, rope_theta);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      4 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q_.y = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(float4& q,  // NOLINT
                                              float4& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
  const auto coef0 = rotary_embedding_coefficient(
      4 * tid, rot_embed_dim, float_t_step, rope_theta);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  k_.x = rotary_embedding_transform(k_.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      4 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q_.y = rotary_embedding_transform(q_.y, coef1);
  k_.y = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step, rope_theta);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,  // NOLINT
                                              uint32_t& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef = rotary_embedding_coefficient(
      2 * tid, rot_embed_dim, float_t_step, rope_theta);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(uint2& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      4 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      4 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint2& q,  // NOLINT
                                              uint2& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      4 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      4 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      8 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      8 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 = rotary_embedding_coefficient(
      8 * tid + 4, rot_embed_dim, float_t_step, rope_theta);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 = rotary_embedding_coefficient(
      8 * tid + 6, rot_embed_dim, float_t_step, rope_theta);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(uint4& q,  // NOLINT
                                              uint4& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      8 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      8 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 = rotary_embedding_coefficient(
      8 * tid + 4, rot_embed_dim, float_t_step, rope_theta);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 = rotary_embedding_coefficient(
      8 * tid + 6, rot_embed_dim, float_t_step, rope_theta);
  q.w = rotary_embedding_transform(q.w, coef3);
  k.w = rotary_embedding_transform(k.w, coef3);
}

#ifdef ENABLE_BF16
inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,      // NOLINT
                                              __nv_bfloat162& k,      // NOLINT
                                              __nv_bfloat162& cos,    // NOLINT
                                              __nv_bfloat162& sin) {  // NOLINT
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,  // NOLINT
                                              __nv_bfloat162& k,  // NOLINT
                                              float2& cos,        // NOLINT
                                              float2& sin) {      // NOLINT
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,      // NOLINT
                                              bf16_4_t& k,      // NOLINT
                                              bf16_4_t& cos,    // NOLINT
                                              bf16_4_t& sin) {  // NOLINT
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,    // NOLINT
                                              bf16_4_t& k,    // NOLINT
                                              float4& cos,    // NOLINT
                                              float4& sin) {  // NOLINT
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  q.x = rotary_embedding_transform(q.x, cos_.x, sin_.x);
  k.x = rotary_embedding_transform(k.x, cos_.x, sin_.x);
  q.y = rotary_embedding_transform(q.y, cos_.y, sin_.y);
  k.y = rotary_embedding_transform(k.y, cos_.y, sin_.y);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,      // NOLINT
                                              bf16_8_t& k,      // NOLINT
                                              bf16_8_t& cos,    // NOLINT
                                              bf16_8_t& sin) {  // NOLINT
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,     // NOLINT
                                              bf16_8_t& k,     // NOLINT
                                              Float8_& cos,    // NOLINT
                                              Float8_& sin) {  // NOLINT
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef = rotary_embedding_coefficient(
      2 * tid, rot_embed_dim, float_t_step, rope_theta);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,  // NOLINT
                                              __nv_bfloat162& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef = rotary_embedding_coefficient(
      2 * tid, rot_embed_dim, float_t_step, rope_theta);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      4 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      4 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,  // NOLINT
                                              bf16_4_t& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      4 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      4 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      8 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      8 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 = rotary_embedding_coefficient(
      8 * tid + 4, rot_embed_dim, float_t_step, rope_theta);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 = rotary_embedding_coefficient(
      8 * tid + 6, rot_embed_dim, float_t_step, rope_theta);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,  // NOLINT
                                              bf16_8_t& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float inv_compression_ratio,
                                              float rope_theta) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step *= inv_compression_ratio;
  const auto coef0 = rotary_embedding_coefficient(
      8 * tid, rot_embed_dim, float_t_step, rope_theta);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 = rotary_embedding_coefficient(
      8 * tid + 2, rot_embed_dim, float_t_step, rope_theta);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 = rotary_embedding_coefficient(
      8 * tid + 4, rot_embed_dim, float_t_step, rope_theta);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 = rotary_embedding_coefficient(
      8 * tid + 6, rot_embed_dim, float_t_step, rope_theta);
  q.w = rotary_embedding_transform(q.w, coef3);
  k.w = rotary_embedding_transform(k.w, coef3);
}
#endif  // ENABLE_BF16

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
template <typename T>
struct K_vec_acum_fp32_ {};

template <>
struct K_vec_acum_fp32_<uint32_t> {
  using Type = float2;
};
#endif

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template <typename T>
struct V_vec_acum_fp32_ {};
// template <> struct V_vec_acum_fp32_<float>  { using Type = float;  };
// template <> struct V_vec_acum_fp32_<float2> { using Type = float2; };
template <>
struct V_vec_acum_fp32_<float4> {
  using Type = float4;
};
// template <> struct V_vec_acum_fp32_<uint32_t> { using Type = float2;   };
// template <> struct V_vec_acum_fp32_<uint2   > { using Type = Float4_;  };
template <>
struct V_vec_acum_fp32_<uint4> {
  using Type = Float8_;
};

#ifdef ENABLE_BF16
template <>
struct V_vec_acum_fp32_<__nv_bfloat162> {
  using Type = float2;
};
template <>
struct V_vec_acum_fp32_<bf16_4_t> {
  using Type = Float4_;
};
template <>
struct V_vec_acum_fp32_<bf16_8_t> {
  using Type = Float8_;
};
#endif  // ENABLE_BF16

#endif

// clang-format on

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(float& dst, float src) {  // NOLINT
  dst = src;
}

inline __device__ void convert_from_float(float4& dst, float4 src) {  // NOLINT
  dst = src;
}

inline __device__ void convert_from_float(phi::float16& dst,  // NOLINT
                                          float src) {
  dst = static_cast<phi::float16>(src);
}

inline __device__ void convert_from_float(uint4& dst, Float8_ src) {  // NOLINT
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16& dst,  // NOLINT
                                          float src) {         // NOLINT
  dst = __float2bfloat16(src);
}

inline __device__ void convert_from_float(__nv_bfloat162& dst,  // NOLINT
                                          float2 src) {         // NOLINT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  dst = __float22bfloat162_rn(src);
#else
  dst = __floats2bfloat162_rn(src.x, src.y);
#endif
}

inline __device__ void convert_from_float(bf16_4_t& dst,  // NOLINT
                                          Float4_ src) {  // NOLINT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
#else
  dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
  dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void convert_from_float(bf16_4_t& dst,  // NOLINT
                                          float4 src) {   // NOLINT
  convert_from_float(
      dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

inline __device__ void convert_from_float(bf16_8_t& dst,  // NOLINT
                                          Float8_ src) {  // NOLINT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
  dst.z = __float22bfloat162_rn(src.z);
  dst.w = __float22bfloat162_rn(src.w);
#else
  dst.x = __floats2bfloat162_rn(src.x.x, src.x.y);
  dst.y = __floats2bfloat162_rn(src.y.x, src.y.y);
  dst.z = __floats2bfloat162_rn(src.z.x, src.z.y);
  dst.w = __floats2bfloat162_rn(src.w.x, src.w.y);
#endif
}
#endif  // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void zero(uint16_t& dst) { dst = uint16_t(0); }  // NOLINT

template <typename T>
inline __device__ void zero(T& dst) {  // NOLINT
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;
#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

template <int WARPS_PER_BLOCK, int WARP_SIZE_T = 32>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / WARP_SIZE_T;
  int lane = threadIdx.x % WARP_SIZE_T;

#pragma unroll
  for (int mask = WARP_SIZE_T / 2; mask >= 1; mask /= 2) {
#ifdef PADDLE_WITH_HIP
    sum += __shfl_xor(sum, mask);
#else
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
#endif
  }

  if (lane == 0) {
    red_smem[warp] = sum;
  }
  __syncthreads();

  if (lane < WARPS_PER_BLOCK) {
    sum = red_smem[lane];
  }

#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
#ifdef PADDLE_WITH_HIP
    sum += __shfl_xor(sum, mask);
#else
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
#endif
  }
#ifdef PADDLE_WITH_HIP
  return __shfl(sum, 0);
#else
  return __shfl_sync(uint32_t(-1), sum, 0);
#endif
}

template <typename T>
__device__ __inline__ T ClipFunc(const T v, const T min, const T max) {
  if (v > max) return max;
  if (v < min) return min;
  return v;
}

template <typename InType, typename OutType>
__forceinline__ __device__ OutType QuantHelperFunc(const InType input,
                                                   const float scale,
                                                   const int round_type,
                                                   const float max_bound,
                                                   const float min_bound) {
  float quant_value = max_bound * scale * input;

  if (round_type == 0) {
    quant_value = static_cast<float>(rint(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <typename T, typename LoadT = T>
struct MMHALoad {
  explicit MMHALoad(const LoadT* src) : src_(src) {}

  template <typename Vec>
  __device__ void load(Vec& dst, int idx) {  // NOLINT
    dst = *reinterpret_cast<const Vec*>(src_ + idx);
  }

  const LoadT* src_;
};

template <typename T, typename StoreT = T, bool Smooth = false>
struct MMHAStore {
  explicit MMHAStore(StoreT* dst) : dst_(dst) {}

  template <typename Vec>
  __device__ void store(Vec& src, size_t idx) {
    *reinterpret_cast<Vec*>(dst_ + idx) = src;
  }

  template <typename Vec>
  __device__ void store(const Vec& src, float scale, size_t idx) {
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using TVec = phi::AlignedVector<T, VecSize>;
    TVec src_vec;
    *reinterpret_cast<Vec*>(&src_vec) = src;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src_vec[i] = static_cast<T>(static_cast<float>(src_vec[i]) * scale);
    }
    phi::Store<T, VecSize>(src_vec, dst_ + idx);
  }

  StoreT* dst_;
};

template <typename T>
struct MMHAStore<T, T, true> {
  MMHAStore(T* dst, const T* shift, const T* smooth, const int cols)
      : dst_(dst), shift_(shift), smooth_(smooth), cols_(cols) {}

  template <typename Vec>
  __device__ void store(Vec& src, int idx) {  // NOLINT
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using TVec = phi::AlignedVector<T, VecSize>;
    TVec src_vec;
    TVec shift_vec;
    TVec smooth_vec;

    *reinterpret_cast<Vec*>(&src_vec) = src;
    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src_vec[i] = (src_vec[i] + shift_vec[i]) * smooth_vec[i];
    }

    phi::Store<T, VecSize>(src_vec, dst_ + idx);
  }

  T* dst_;
  const T* shift_;
  const T* smooth_;
  const int cols_;
};

template <typename T>
struct MMHALoad<T, int32_t> {
  MMHALoad(const int32_t* src, const float* dequant_scales, const int cols)
      : src_(src), dequant_scales_(dequant_scales), cols_(cols) {}

  template <typename Vec>
  __device__ void load(Vec& dst, int idx) {  // NOLINT
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using SrcVec = phi::AlignedVector<int32_t, VecSize>;
    using DstVec = phi::AlignedVector<T, VecSize>;
    using ScaleVec = phi::AlignedVector<float, VecSize>;

    SrcVec src_vec;
    DstVec dst_vec;
    ScaleVec scale_vec;

    phi::Load<int32_t, VecSize>(src_ + idx, &src_vec);
    phi::Load<float, VecSize>(dequant_scales_ + idx % cols_, &scale_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      dst_vec[i] =
          static_cast<T>(static_cast<float>(src_vec[i]) * scale_vec[i]);
    }
    dst = *reinterpret_cast<Vec*>(&dst_vec);
  }

  const int32_t* src_;
  const float* dequant_scales_;
  const int cols_;
};

template <typename T>
struct MMHAStore<T, int8_t> {
  MMHAStore(int8_t* dst,
            const int quant_round_type,
            const float quant_scale,
            const float quant_max_bound,
            const float quant_min_bound)
      : dst_(dst),
        quant_round_type_(quant_round_type),
        quant_scale_(quant_scale),
        quant_max_bound_(quant_max_bound),
        quant_min_bound_(quant_min_bound) {}

  template <typename Vec>
  __device__ void store(Vec& src, int idx) {  // NOLINT
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using SrcVec = phi::AlignedVector<T, VecSize>;
    using DstVec = phi::AlignedVector<int8_t, VecSize>;

    SrcVec src_vec;
    *reinterpret_cast<Vec*>(&src_vec) = src;

    DstVec dst_vec;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      dst_vec[i] =
          QuantHelperFunc<float, int8_t>(static_cast<float>(src_vec[i]),
                                         quant_scale_,
                                         quant_round_type_,
                                         quant_max_bound_,
                                         quant_min_bound_);
    }

    phi::Store<int8_t, VecSize>(dst_vec, dst_ + idx);
  }

  int8_t* dst_;
  const int quant_round_type_;
  const float quant_scale_;
  const float quant_max_bound_;
  const float quant_min_bound_;
};

template <typename T>
struct MMHAStore<T, int8_t, true> {
  MMHAStore(int8_t* dst,
            const T* shift,
            const T* smooth,
            const int cols,
            const int quant_round_type,
            const float quant_scale,
            const float quant_max_bound,
            const float quant_min_bound)
      : dst_(dst),
        quant_round_type_(quant_round_type),
        quant_scale_(quant_scale),
        quant_max_bound_(quant_max_bound),
        quant_min_bound_(quant_min_bound),
        shift_(shift),
        smooth_(smooth),
        cols_(cols) {}

  template <typename Vec>
  __device__ void store(Vec& src, int idx) {  // NOLINT
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using SrcVec = phi::AlignedVector<T, VecSize>;
    using DstVec = phi::AlignedVector<int8_t, VecSize>;

    SrcVec src_vec;
    DstVec dst_vec;
    SrcVec shift_vec;
    SrcVec smooth_vec;

    *reinterpret_cast<Vec*>(&src_vec) = src;
    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src_vec[i] = (src_vec[i] + shift_vec[i]) * smooth_vec[i];
      dst_vec[i] =
          QuantHelperFunc<float, int8_t>(static_cast<float>(src_vec[i]),
                                         quant_scale_,
                                         quant_round_type_,
                                         quant_max_bound_,
                                         quant_min_bound_);
    }

    phi::Store<int8_t, VecSize>(dst_vec, dst_ + idx);
  }

  int8_t* dst_;
  const T* shift_;
  const T* smooth_;
  const int cols_;
  const int quant_round_type_;
  const float quant_scale_;
  const float quant_max_bound_;
  const float quant_min_bound_;
};

template <int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N],
                                const K_vec (&k)[N],
                                float inv_sqrt_dh) {
  K_vec inv_q = mul<K_vec, K_vec, float>(q[0], inv_sqrt_dh);
  K_vec qk_vec = mul<K_vec, K_vec, K_vec>(inv_q, k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    inv_q = mul<K_vec, K_vec, float>(q[ii], inv_sqrt_dh);
    qk_vec = fma(inv_q, k[ii], qk_vec);
  }

  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2) {
#ifdef PADDLE_WITH_HIP
    qk += __shfl_xor(qk, mask);
#else
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
#endif
  }
  return qk;
}

template <typename T, int THREADS_PER_KEY>
struct Qk_dot {
  template <typename K_vec, int N>
  static inline __device__ float dot(const K_vec (&q)[N],
                                     const K_vec (&k)[N],
                                     float inv_sqrt_dh) {
    return qk_dot_<THREADS_PER_KEY>(q, k, inv_sqrt_dh);
  }
};

constexpr int32_t WARP_SIZE_TMP = 32;
constexpr int32_t HALF_WARP_TMP = 16;
constexpr float QUANT_MAX_BOUND = 127.0;
constexpr float QUANT_MIN_BOUND = -127.0;

template <typename T>
struct QuantFunc {
  __host__ __device__ uint8_t operator()(T x, float quant_scale) {
#ifdef PADDLE_WITH_HIP
    float tmp;
    if constexpr (kernel_dtype_is_same<T, half>::value) {
      tmp = __half2float(x) * quant_scale;
    } else {
      tmp = static_cast<float>(x) * quant_scale;
    }
#else
    float tmp = static_cast<float>(x) * quant_scale;
#endif
    tmp = round(tmp);
    if (tmp > QUANT_MAX_BOUND)
      tmp = QUANT_MAX_BOUND;
    else if (tmp < QUANT_MIN_BOUND)
      tmp = QUANT_MIN_BOUND;
    return static_cast<uint8_t>(tmp + 128.0f);
  }
};

template <typename T>
struct MaxFunc {
  __device__ T operator()(T a, T b) { return max(a, b); }
};

template <>
struct MaxFunc<half> {
  __device__ half operator()(half a, half b) {
#if __CUDA_ARCH__ >= 800 || defined(PADDLE_WITH_HIP)
    return __hmax(a, b);
#else
    return max(static_cast<float>(a), static_cast<float>(b));
#endif
  }
};

#if CUDA_VERSION >= 11000 && defined(ENABLE_BF16)
template <>
struct MaxFunc<__nv_bfloat16> {
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
    return __hmax(a, b);
#else
    return max(static_cast<float>(a), static_cast<float>(b));
#endif
  }
};
#endif

template <typename T>
struct AbsFunc {
  __device__ T operator()(T x) { return abs(x); }
};

template <>
struct AbsFunc<half> {
  __device__ half operator()(half x) {
#if __CUDA_ARCH__ >= 800 || defined(PADDLE_WITH_HIP)
    return __habs(x);
#else
    return abs(static_cast<float>(x));
#endif
  }
};

#if CUDA_VERSION >= 11000 && defined(ENABLE_BF16)
template <>
struct AbsFunc<__nv_bfloat16> {
  __device__ __nv_bfloat16 operator()(__nv_bfloat16 x) {
#if __CUDA_ARCH__ >= 800
    return __habs(x);
#else
    return abs(static_cast<float>(x));
#endif
  }
};
#endif

template <typename T, typename Vec, int VecSize>
__inline__ __device__ T LocalReduceMax(Vec& vec) {  // NOLINT
#ifdef PADDLE_WITH_HIP
  T local_max;
  if constexpr (kernel_dtype_is_same<T, half>::value) {
    local_max = __float2half(0.0f);
  } else {
    local_max = static_cast<T>(0.0f);
  }
#else
  T local_max = static_cast<T>(0.0);
#endif
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    local_max = vec[i] > local_max ? vec[i] : local_max;
  }
  return local_max;
}

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
#pragma unroll
  for (int mask = HALF_WARP_TMP; mask > 0; mask >>= 1) {
#ifdef PADDLE_WITH_HIP
    val = MaxFunc<T>()(val, __shfl_xor(val, mask, WARP_SIZE_TMP));
#else
    val =
        MaxFunc<T>()(val, __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE_TMP));
#endif
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceAbsMax(T val, unsigned mask) {
  static __shared__ T smem[WARP_SIZE_TMP];
  int32_t lane_id = threadIdx.x % WARP_SIZE_TMP;
  int32_t warp_id = threadIdx.x / WARP_SIZE_TMP;

  val = WarpReduceAbsMax(val, mask);

  if (lane_id == 0) {
    smem[warp_id] = val;
  }

  __syncthreads();

#ifdef PADDLE_WITH_HIP
  T abs_max_val;
  if constexpr (kernel_dtype_is_same<T, half>::value) {
    abs_max_val = __float2half(0.0f);
  } else {
    abs_max_val = static_cast<T>(0.0f);
  }
  abs_max_val = smem[threadIdx.x];
#else
  T abs_max_val = (threadIdx.x < (blockDim.x / WARP_SIZE_TMP))
                      ? smem[threadIdx.x]
                      : static_cast<T>(0.0f);
#endif
  abs_max_val = WarpReduceAbsMax(abs_max_val, mask);
  return abs_max_val;
}

}  // namespace fusion
}  // namespace phi
