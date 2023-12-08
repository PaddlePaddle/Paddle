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

#pragma once

#ifndef MMHA_UTIL_CU_H_
#define MMHA_UTIL_CU_H_
#ifdef __NVCC__

#include <cuda_fp16.h>
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include <float.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/phi/kernels/funcs/fused_gemm_epilogue.h"

// #include
// "paddle/fluid/operators/fused/cutlass/cutlass_extensions/interleaved_numeric_conversion.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/fused/datatype_traits.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

#if defined(__CUDACC__) && CUDA_VERSION >= 11000
#define ENABLE_BF16
#include <cuda_bf16.h>
#endif


namespace paddle {
namespace operators {

namespace {  // NOLINT
namespace plat = paddle::platform;
using float16 = plat::float16;
// using float16 = half;
using bfloat16 = plat::bfloat16;
// using bfloat16 = __nv_bfloat16;

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
#endif // ENABLE_BF16

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
// lyq::todo float8
// lyq::todo uint16_t
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
#endif // ENABLE_BF16

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
#endif // ENABLE_BF16
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

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // Is it better?
    float zero = 0.f;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(zero), "f"(f));
#else
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#endif
  return tmp.u16[0];
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
inline __device__ void mul_pointer_v2(float4* c, float a, uint8_t* b) {
  c->x = a * (static_cast<float>(b[0]) - 128.0);
  c->y = a * (static_cast<float>(b[1]) - 128.0);
  c->z = a * (static_cast<float>(b[2]) - 128.0);
  c->w = a * (static_cast<float>(b[3]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2(float4* c, float a, uint32_t* b) {
  uint8_t* b_tmp = reinterpret_cast<uint8_t*>(b);
  c->x = a * (static_cast<float>(b_tmp[0]) - 128.0);
  c->y = a * (static_cast<float>(b_tmp[1]) - 128.0);
  c->z = a * (static_cast<float>(b_tmp[2]) - 128.0);
  c->w = a * (static_cast<float>(b_tmp[3]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2(float2* c, float a, uint8_t* b) {
  c->x = a * (static_cast<float>(b[0]) - 128.0);
  c->y = a * (static_cast<float>(b[1]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2(float* c, float a, uint8_t* b) {
  c[0] = a * (static_cast<float>(b[0]) - 128.0);
}

template <>
inline __device__ void mul_pointer_v2(uint32_t* c, float a, uint8_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(b[i]) - offset);
  }
}

template <>
inline __device__ void mul_pointer_v2(uint2* c, float a, uint8_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(b[i]) - offset);
  }
}

template <>
inline __device__ void mul_pointer_v2(uint4* c, float a, uint8_t* b) {
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(b[i]) - offset);
  }
}

template <>
inline __device__ void mul_pointer_v2(uint4* c, float a, uint64_t* b) {
  uint8_t* tmp_b = reinterpret_cast<uint8_t*>(b);
  float16* tmp_fp16 = reinterpret_cast<float16*>(c);
  float16 a_prime = static_cast<float16>(a);
  float16 offset = static_cast<float16>(128.0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tmp_fp16[i] = a_prime * (static_cast<float16>(tmp_b[i]) - offset);
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
inline __device__ void mul_pointer_v2(__nv_bfloat162* c, float a, uint8_t* b) {
#if __CUDA_ARCH__ >= 800
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(b)));
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    c_prime[i] *= a_prime;
  }
#endif
}

template <>
inline __device__ void mul_pointer_v2(__nv_bfloat162* c, float a, uint16_t* b) {
#if __CUDA_ARCH__ >= 800
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
#endif
}

template <>
inline __device__ void mul_pointer_v2(bf16_4_t* c, float a, uint8_t* b) {
#if __CUDA_ARCH__ >= 800
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *reinterpret_cast<uint32_t*>(b));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c_prime[i] *= a_prime;
  }
#endif
}

template <>
inline __device__ void mul_pointer_v2(bf16_4_t* c, float a, uint32_t* b) {
#if __CUDA_ARCH__ >= 800
  __nv_bfloat16 a_prime = static_cast<__nv_bfloat16>(a);
  __nv_bfloat16* c_prime = reinterpret_cast<__nv_bfloat16*>(c);
  convert_(c_prime, *b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c_prime[i] *= a_prime;
  }
#endif
}

template <>
inline __device__ void mul_pointer_v2(bf16_8_t* c, float a, uint8_t* b) {
  bf16_4_t* tmp_c = reinterpret_cast<bf16_4_t*>(c);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    mul_pointer_v2<bf16_4_t>(tmp_c + i, a, b + 4 * i);
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
#endif  // ENABLE_BF16


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
#endif // ENABLE_BF16

inline __device__ float2 rotary_embedding_coefficient(const int zid,
                                                      const int rot_embed_dim,
                                                      const float t_step) {
  const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
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

inline __device__ void apply_rotary_embedding(float& q,
                                              float& k,
                                              float& cos,
                                              float& sin) {
  return;
}

inline __device__ void apply_rotary_embedding(float2& q,
                                              float2& k,
                                              float2& cos,
                                              float2& sin) {
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(float4& q,
                                              float4& k,
                                              float4& cos,
                                              float4& sin) {
  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  q_.x = rotary_embedding_transform(q_.x, cos_.x, sin_.x);
  k_.x = rotary_embedding_transform(k_.x, cos_.x, sin_.x);
  q_.y = rotary_embedding_transform(q_.y, cos_.y, sin_.y);
  k_.y = rotary_embedding_transform(k_.y, cos_.y, sin_.y);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,
                                              uint32_t& k,
                                              uint32_t& cos,
                                              uint32_t& sin) {
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,
                                              uint32_t& k,
                                              float2& cos,
                                              float2& sin) {
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(uint2& q,
                                              uint2& k,
                                              uint2& cos,
                                              uint2& sin) {
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.x);
}

inline __device__ void apply_rotary_embedding(uint2& q,
                                              uint2& k,
                                              float4& cos,
                                              float4& sin) {
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  q.x = rotary_embedding_transform(q.x, cos_.x, sin_.x);
  k.x = rotary_embedding_transform(k.x, cos_.x, sin_.x);
  q.y = rotary_embedding_transform(q.y, cos_.y, sin_.y);
  k.y = rotary_embedding_transform(k.y, cos_.y, sin_.x);
}

inline __device__ void apply_rotary_embedding(uint4& q,
                                              uint4& k,
                                              uint4& cos,
                                              uint4& sin) {
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(uint4& q,
                                              uint4& k,
                                              Float8_& cos,
                                              Float8_& sin) {
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(
    float& q, int zid, int rot_embed_dim, int t_step, float compression_ratio) {
  return;
}

inline __device__ void apply_rotary_embedding(float& q,
                                              float& k,
                                              int zid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  return;
}

inline __device__ void apply_rotary_embedding(float2& q,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, float_t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float2& q,
                                              float2& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, float_t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, float_t_step);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, float_t_step);
  q_.y = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(float4& q,
                                              float4& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
  Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, float_t_step);
  q_.x = rotary_embedding_transform(q_.x, coef0);
  k_.x = rotary_embedding_transform(k_.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, float_t_step);
  q_.y = rotary_embedding_transform(q_.y, coef1);
  k_.y = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint32_t& q,
                                              uint32_t& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, float_t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(
    uint2& q, int tid, int rot_embed_dim, int t_step, float compression_ratio) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint2& q,
                                              uint2& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(
    uint4& q, int tid, int rot_embed_dim, int t_step, float compression_ratio) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, float_t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, float_t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(uint4& q,
                                              uint4& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, float_t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, float_t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
  k.w = rotary_embedding_transform(k.w, coef3);
}

#ifdef ENABLE_BF16
inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,
                                              __nv_bfloat162& k,
                                              __nv_bfloat162& cos,
                                              __nv_bfloat162& sin) {
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,
                                              __nv_bfloat162& k,
                                              float2& cos,
                                              float2& sin) {
  q = rotary_embedding_transform(q, cos, sin);
  k = rotary_embedding_transform(k, cos, sin);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,
                                              bf16_4_t& k,
                                              bf16_4_t& cos,
                                              bf16_4_t& sin) {
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,
                                              bf16_4_t& k,
                                              float4& cos,
                                              float4& sin) {
  Float4_& cos_ = *reinterpret_cast<Float4_*>(&cos);
  Float4_& sin_ = *reinterpret_cast<Float4_*>(&sin);
  q.x = rotary_embedding_transform(q.x, cos_.x, sin_.x);
  k.x = rotary_embedding_transform(k.x, cos_.x, sin_.x);
  q.y = rotary_embedding_transform(q.y, cos_.y, sin_.y);
  k.y = rotary_embedding_transform(k.y, cos_.y, sin_.y);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,
                                              bf16_8_t& k,
                                              bf16_8_t& cos,
                                              bf16_8_t& sin) {
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,
                                              bf16_8_t& k,
                                              Float8_& cos,
                                              Float8_& sin) {
  q.x = rotary_embedding_transform(q.x, cos.x, sin.x);
  k.x = rotary_embedding_transform(k.x, cos.x, sin.x);
  q.y = rotary_embedding_transform(q.y, cos.y, sin.y);
  k.y = rotary_embedding_transform(k.y, cos.y, sin.y);
  q.z = rotary_embedding_transform(q.z, cos.z, sin.z);
  k.z = rotary_embedding_transform(k.z, cos.z, sin.z);
  q.w = rotary_embedding_transform(q.w, cos.w, sin.w);
  k.w = rotary_embedding_transform(k.w, cos.w, sin.w);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, float_t_step);
  q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(__nv_bfloat162& q,
                                              __nv_bfloat162& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (2 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef =
      rotary_embedding_coefficient(2 * tid, rot_embed_dim, float_t_step);
  q = rotary_embedding_transform(q, coef);
  k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q,
                                              bf16_4_t& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (4 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(4 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, float_t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, float_t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q,
                                              bf16_8_t& k,
                                              int tid,
                                              int rot_embed_dim,
                                              int t_step,
                                              float compression_ratio) {
  if (8 * tid >= rot_embed_dim) {
    return;
  }
  float float_t_step = static_cast<float>(t_step);
  float_t_step /= compression_ratio;
  const auto coef0 =
      rotary_embedding_coefficient(8 * tid, rot_embed_dim, float_t_step);
  q.x = rotary_embedding_transform(q.x, coef0);
  k.x = rotary_embedding_transform(k.x, coef0);
  const auto coef1 =
      rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, float_t_step);
  q.y = rotary_embedding_transform(q.y, coef1);
  k.y = rotary_embedding_transform(k.y, coef1);
  const auto coef2 =
      rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, float_t_step);
  q.z = rotary_embedding_transform(q.z, coef2);
  k.z = rotary_embedding_transform(k.z, coef2);
  const auto coef3 =
      rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, float_t_step);
  q.w = rotary_embedding_transform(q.w, coef3);
  k.w = rotary_embedding_transform(k.w, coef3);
}
#endif  // ENABLE_BF16

}  // namespace

}  // namespace operators
}  // namespace paddle
#endif
