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

#ifndef PADDLE_WITH_HIP
#pragma once

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <cuda_fp16.h>
#include <float.h>
#include <cub/cub.cuh>

#include "paddle/phi/common/datatype_traits.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace fusion {

#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#define MMHA_USE_FP32_ACUM_FOR_OUT
#define MMHA_USE_FP32_ACUM_FOR_FMA

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

#ifdef ENABLE_BF16
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
#endif  // ENABLE_BF16

//------------------------------------
template <typename T, int THREADS_PER_KEY>
struct K_vec_I_ {
  using Type = uint8_t;
};

template <>
struct K_vec_I_<float16, 4> {
  using Type = uint16_t;
};

template <>
struct K_vec_I_<float16, 2> {
  using Type = uint32_t;
};

template <>
struct K_vec_I_<float16, 1> {
  using Type = uint64_t;
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
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
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
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
  return c;
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
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

template <typename T, typename IntT>
inline __device__ void mul_pointer_v2(T* c, float a, IntT* b);

template <>
inline __device__ void mul_pointer_v2(float4* c, float a, uint32_t* b) {
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x = a * (static_cast<float>(b_tmp[0]) - 128.0);
  c->y = a * (static_cast<float>(b_tmp[1]) - 128.0);
  c->z = a * (static_cast<float>(b_tmp[2]) - 128.0);
  c->w = a * (static_cast<float>(b_tmp[3]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2(float* c, float a, uint8_t* b) {
  c[0] = a * (static_cast<float>(b[0]) - 128.0);
}

inline __device__ void convert_(float16* result, uint32_t const& source) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  uint32_t* h = reinterpret_cast<uint32_t*>(result);
  uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
#endif
}

// float16 * 2 <- uint8_t * 2
template <>
inline __device__ void mul_pointer_v2(uint32_t* c, float a, uint16_t* b) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  uint32_t tmp_uint32 = 0;
  uint32_t* h = &tmp_uint32;
  uint16_t tmp_b = *b;
  uint32_t i8s = *reinterpret_cast<uint32_t*>(&tmp_b);

  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));

  half2 tmp_half2 = *reinterpret_cast<half2*>(h);
  tmp_half2.x *= static_cast<half>(a);
  tmp_half2.y *= static_cast<half>(a);

  c[0] = *reinterpret_cast<uint32_t*>(&tmp_half2);

#endif
}

// float16 * 4 <- uint8_t * 4
template <>
inline __device__ void mul_pointer_v2(uint2* c, float a, uint32_t* b) {
  float16* c_prime = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  convert_(c_prime, *b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c_prime[i] *= a_prime;
  }
}
// float16 * 8 <- uint8_t * 8
template <>
inline __device__ void mul_pointer_v2(uint4* c, float a, uint64_t* b) {
  uint2* tmp_c = reinterpret_cast<uint2*>(c);
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(b);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    mul_pointer_v2(tmp_c + i, a, tmp_b + i);
  }
}

#ifdef ENABLE_BF16
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
inline __device__ void mul_pointer_v2(__nv_bfloat162* c, float a, uint16_t* b) {
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
  c->x = c->x * scale;
  c->y = c->y * scale;
}

template <>
inline __device__ void mul_pointer_v2(bf16_4_t* c, float a, uint32_t* b) {
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c_prime[i] = c_prime[i] * a_prime;
  }
}

template <>
inline __device__ void mul_pointer_v2(bf16_8_t* c, float a, uint64_t* b) {
  bf16_4_t* tmp_c = reinterpret_cast<bf16_4_t*>(c);
  uint64_t bb = *b;
  uint32_t* tmp_b = reinterpret_cast<uint32_t*>(&bb);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    mul_pointer_v2<bf16_4_t>(tmp_c + i, a, tmp_b + i);
  }
}
#endif  // ENABLE_BF16

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
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
  return c;
}

template <>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
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
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
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
  asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
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

template <typename T, typename D>
inline __device__ T round_tmp(D val);

template <>
inline __device__ uint8_t round_tmp(float val) {
  float quant_value = roundWithTiesToEven(val);
  quant_value = quant_value > 127.0f ? 127.0f : quant_value;
  quant_value = quant_value < -127.0f ? -127.0f : quant_value;
  return static_cast<uint8_t>(quant_value + 128.0);
}

template <>
inline __device__ uint8_t round_tmp(float16 val) {
  float quant_value = roundWithTiesToEven(static_cast<float>(val));
  quant_value = quant_value > 127.0f ? 127.0f : quant_value;
  quant_value = quant_value < -127.0f ? -127.0f : quant_value;
  return static_cast<uint8_t>(quant_value + 128.0);
}

#ifdef ENABLE_BF16
template <>
inline __device__ uint8_t round_tmp(__nv_bfloat16 val) {
  float quant_value =
      static_cast<float>(roundWithTiesToEven(static_cast<float>(val)));
  quant_value = quant_value > 127.0f ? 127.0f : quant_value;
  quant_value = quant_value < -127.0f ? -127.0f : quant_value;
  return static_cast<uint8_t>(quant_value + 128.0);
}
#endif

template <>
inline __device__ uint16_t round_tmp(float2 val) {
  union {
    uint16_t ret;
    uint8_t tmp[2];
  };
  tmp[0] = round_tmp<uint8_t, float>(val.x);
  tmp[1] = round_tmp<uint8_t, float>(val.y);
  return ret;
}

template <>
inline __device__ uint32_t round_tmp(float4 val) {
  union {
    uint32_t ret;
    uint8_t tmp[4];
  };
  tmp[0] = round_tmp<uint8_t, float>(val.x);
  tmp[1] = round_tmp<uint8_t, float>(val.y);
  tmp[2] = round_tmp<uint8_t, float>(val.z);
  tmp[3] = round_tmp<uint8_t, float>(val.w);
  return ret;
}

template <>
inline __device__ uint16_t round_tmp(uint32_t val) {
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
    int8[i] = round_tmp<uint8_t, float16>(fp16[i]);
  }

  return ret;
}

template <>
inline __device__ uint32_t round_tmp(uint2 val) {
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
    int8[i] = round_tmp<uint8_t, float16>(tmp_fp16[i]);
  }
  return ret;
}

template <>
inline __device__ uint64_t round_tmp(uint4 val) {
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
    int8[i] = round_tmp<uint8_t, float16>(tmp_fp16[i]);
  }
  return ret;
}

#ifdef ENABLE_BF16
template <>
inline __device__ uint16_t round_tmp(__nv_bfloat162 val) {
  union {
    uint8_t tmp[2];
    uint16_t ret;
  };
  tmp[0] = round_tmp<uint8_t, __nv_bfloat16>(val.x);
  tmp[1] = round_tmp<uint8_t, __nv_bfloat16>(val.y);
  return ret;
}

template <>
inline __device__ uint32_t round_tmp(bf16_4_t val) {
  union {
    uint16_t tmp[2];
    uint32_t ret;
  };
  tmp[0] = round_tmp<uint16_t, __nv_bfloat162>(val.x);
  tmp[1] = round_tmp<uint16_t, __nv_bfloat162>(val.y);
  return ret;
}

template <>
inline __device__ uint64_t round_tmp(bf16_8_t val) {
  union {
    uint16_t int16[4];
    uint64_t int64;
  };
  int16[0] = round_tmp<uint16_t, __nv_bfloat162>(val.x);
  int16[1] = round_tmp<uint16_t, __nv_bfloat162>(val.y);
  int16[2] = round_tmp<uint16_t, __nv_bfloat162>(val.z);
  int16[3] = round_tmp<uint16_t, __nv_bfloat162>(val.w);
  return int64;
}
#endif

inline __device__ float2 rotary_embedding_coefficient(const int zid,
                                                      const int rot_embed_dim,
                                                      const float t_step) {
  const float inv_freq =
      t_step / pow(10000.0f, zid / static_cast<float>(rot_embed_dim));
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
  k.y = rotary_embedding_transform(k.y, cos.y, sin.x);
}

inline __device__ void apply_rotary_embedding(
    uint2& q,       // NOLINT equals 4 half.
    uint2& k,       // NOLINT
    float4& cos,    // NOLINT 2 float2 cos.
    float4& sin) {  // NOLINT
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  // cos_.x is float2
  q.x = rotary_embedding_transform(q.x, cos_.x, sin_.x);
  k.x = rotary_embedding_transform(k.x, cos_.x, sin_.x);
  q.y = rotary_embedding_transform(q.y, cos_.y, sin_.y);
  k.y = rotary_embedding_transform(k.y, cos_.y, sin_.y);
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
                                              int t_step) {
  return;
}

inline __device__ void apply_rotary_embedding(
    float& q, float& k, int zid, int rot_embed_dim, int t_step) {  // NOLINT
  return;
}

inline __device__ void apply_rotary_embedding(float2& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(
    float2& q, float2& k, int tid, int rot_embed_dim, int t_step) {  // NOLINT
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }

  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q_.y = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(
    float4& q, float4& k, int tid, int rot_embed_dim, int t_step) {  // NOLINT
  if (4 * tid >= rot_embed_dim) {
    return;
  }

  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  k_.x = rotary_embedding_transform(k_.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q_.y = rotary_embedding_transform(q_.y, coef1);
  k_.y = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,  // NOLINT
                                              uint32_t& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {  // NOLINT
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(uint2& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(
    uint2& q, uint2& k, int tid, int rot_embed_dim, int t_step) {  // NOLINT
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(
    uint4& q, uint4& k, int tid, int rot_embed_dim, int t_step) {  // NOLINT
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
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
                                              int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,  // NOLINT
                                              __nv_bfloat162& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,  // NOLINT
                                              bf16_4_t& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {  // NOLINT
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,  // NOLINT
                                              bf16_8_t& k,  // NOLINT
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step) {  // NOLINT
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
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

template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
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
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  return __shfl_sync(uint32_t(-1), sum, 0);
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
  __device__ void load(Vec& dst, int idx) {
    dst = *reinterpret_cast<const Vec*>(src_ + idx);
  }

  const LoadT* src_;
};

template <typename T, typename StoreT = T, bool Smooth = false>
struct MMHAStore {
  explicit MMHAStore(StoreT* dst) : dst_(dst) {}

  template <typename Vec>
  __device__ void store(Vec& src, int idx) {
    *reinterpret_cast<Vec*>(dst_ + idx) = src;
  }

  StoreT* dst_;
};

template <typename T>
struct MMHAStore<T, T, true> {
  MMHAStore(T* dst, const T* shift, const T* smooth, const int cols)
      : dst_(dst), shift_(shift), smooth_(smooth), cols_(cols) {}

  template <typename Vec>
  __device__ void store(Vec& src, int idx) {
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
  __device__ void load(Vec& dst, int idx) {
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

inline __device__ float4 hmma_fp32_tensorcore(const uint2& a, uint32_t b) {
  float4 c;
  float zero = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
      "    {%0, %1, %2, %3}, \n"
      "    {%4, %5}, \n"
      "    {%6}, \n"
      "    {%7, %7, %7, %7}; \n"

      : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
      : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
  return c;
}

template <int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N],
                                     const uint32_t (&k)[N],
                                     float inv_sqrt_dh) {
#if defined(MMHA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
  using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
  using K_vec_acum = uint32_t;
#endif
  K_vec_acum inv_q = mul<K_vec_acum, uint32_t, float>(q[0], inv_sqrt_dh);
  K_vec_acum qk_vec = mul<K_vec_acum, K_vec_acum, uint32_t>(inv_q, k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    inv_q = mul<K_vec_acum, uint32_t, float>(q[ii], inv_sqrt_dh);
    qk_vec = fma(inv_q, k[ii], qk_vec);
  }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
  uint32_t qk_vec_ = float2_to_half2(qk_vec);
  return hmma_fp32_tensorcore(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
  return hmma_fp32_tensorcore(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
  return 0.f;
#endif
}

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
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
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

template <>
struct Qk_dot<float16, 4> {
  template <int N>
  static inline __device__ float dot(const uint32_t (&q)[N],
                                     const uint32_t (&k)[N],
                                     float inv_sqrt_dh) {
#if defined(MMHA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
    return qk_hmma_dot_(q, k, inv_sqrt_dh);
#else
    return qk_dot_<4>(q, k, inv_sqrt_dh);
#endif
  }
};

constexpr int32_t WARP_SIZE = 32;
constexpr int32_t HALF_WARP = 16;
constexpr float QUANT_MAX_BOUND = 127.0;
constexpr float QUANT_MIN_BOUND = -127.0;

template <typename T>
struct QuantFunc {
  __host__ __device__ uint8_t operator()(T x, float quant_scale) {
    float tmp = static_cast<float>(x) * quant_scale;
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
#if __CUDA_ARCH__ >= 800
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
#if __CUDA_ARCH__ >= 800
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
  T local_max = static_cast<T>(0.0);
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    local_max = vec[i] > local_max ? vec[i] : local_max;
  }
  return local_max;
}

template <typename T>
__inline__ __device__ T WarpReduceAbsMax(T val, unsigned lane_mask) {
#pragma unroll
  for (int mask = HALF_WARP; mask > 0; mask >>= 1) {
    val = MaxFunc<T>()(val, __shfl_xor_sync(lane_mask, val, mask, WARP_SIZE));
  }
  return val;
}

template <typename T>
__inline__ __device__ T BlockReduceAbsMax(T val, unsigned mask) {
  static __shared__ T smem[WARP_SIZE];
  int32_t lane_id = threadIdx.x % WARP_SIZE;
  int32_t warp_id = threadIdx.x / WARP_SIZE;

  val = WarpReduceAbsMax(val, mask);

  if (lane_id == 0) {
    smem[warp_id] = val;
  }

  __syncthreads();

  T abs_max_val = (threadIdx.x < (blockDim.x / WARP_SIZE))
                      ? smem[threadIdx.x]
                      : static_cast<T>(0.0f);
  abs_max_val = WarpReduceAbsMax(abs_max_val, mask);
  return abs_max_val;
}

}  // namespace fusion
}  // namespace phi

#endif  // PADDLE_WITH_HIP
