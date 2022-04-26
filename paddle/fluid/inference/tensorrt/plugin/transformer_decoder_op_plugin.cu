// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdio.h>

#include <cassert>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/inference/tensorrt/plugin/transformer_decoder_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

  #define MMHA_USE_FP32_ACUM_FOR_LOGITS
  #define MMHA_USE_FP32_ACUM_FOR_OUT
  using half = phi::dtype::float16;
  template <typename T>
  struct Masked_multihead_attention_params {
    // output buffer, [B, 1(seq_len), num_head * dim_head]
    T *out;
    // qkv_out, [3, B, 1(seq_len), num_head * dim_head]
    const T *qkv;
    // bias, [3, num_head, dim_head]
    const T *qkv_bias;
    // TODO(wangxi): optimize with input_lengths and max_input_len
    // [bsz, 1, 1, time_step(cache_seq_length)+1]
    const T *attn_mask;
  
    // [2, B, num_head, cache_seq_len(padding max_seq_len), dim_head]
    T *cache_kv;
  
    int batch_size;
    int num_head;
    // int dim_head;
    int timestep;  // cache_seq_length
    int max_seq_length;
  
    // 1.f / sqrt(Dh)
    float inv_sqrt_dh;
  };
  
  struct Float8_ {
    float2 x;
    float2 y;
    float2 z;
    float2 w;
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
  struct Qk_vec_<half, 32> {
    using Type = uint32_t;
  };
  template <>
  struct Qk_vec_<half, 64> {
    using Type = uint32_t;
  };
  template <>
  struct Qk_vec_<half, 128> {
    using Type = uint2;
  };
  
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
  struct K_vec_<half, 4> {
    using Type = uint32_t;
  };
  template <>
  struct K_vec_<half, 2> {
    using Type = uint2;
  };
  template <>
  struct K_vec_<half, 1> {
    using Type = uint4;
  };
  
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
  struct V_vec_<half, 2> {
    using Type = uint32_t;
  };
  template <>
  struct V_vec_<half, 4> {
    using Type = uint2;
  };
  template <>
  struct V_vec_<half, 8> {
    using Type = uint4;
  };
  
  #ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  template <typename T>
  struct V_vec_acum_fp32_ {};
  
  // template<> struct V_vec_acum_fp32_<float   > { using Type = float;    };
  // template<> struct V_vec_acum_fp32_<float2  > { using Type = float2;   };
  template <>
  struct V_vec_acum_fp32_<float4> {
    using Type = float4;
  };
  // template<> struct V_vec_acum_fp32_<uint32_t> { using Type = float2;   };
  // template<> struct V_vec_acum_fp32_<uint2   > { using Type = Float4_;  };
  template <>
  struct V_vec_acum_fp32_<uint4> {
    using Type = Float8_;
  };
  #endif
  
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
  
  template <typename Acc, typename A, typename B>
  inline __device__ Acc mul(A a, B b);
  
  template <>
  inline __device__ float mul<float, float>(float a, float b) {
    return a * b;
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
  
  inline __device__ float cast_to_float(float u) { return u; }
  
  inline __device__ float2 cast_to_float(float2 u) { return u; }
  
  inline __device__ float4 cast_to_float(float4 u) { return u; }
  
  inline __device__ Float8_ cast_to_float(uint4 u) {
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
  }
  
  template <int THREADS_PER_KEY, typename K_vec, int N>
  inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N]) {
    K_vec qk_vec = mul<K_vec, K_vec, K_vec>(q[0], k[0]);
  #pragma unroll
    for (int ii = 1; ii < N; ++ii) {
      qk_vec = fma(q[ii], k[ii], qk_vec);
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
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N]) {
      return qk_dot_<THREADS_PER_KEY>(q, k);
    }
  };
  
  template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
  inline __device__ float block_sum(float *red_smem, float sum) {
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
  
  inline __device__ void convert_from_float(float &dst, float src) {  // NOLINT
    dst = src;
  }
  
  inline __device__ void convert_from_float(float4 &dst, float4 src) {  // NOLINT
    dst = src;
  }
  
  inline __device__ void convert_from_float(half &dst,  // NOLINT
                                            float src) {
    dst = static_cast<half>(src);
  }
  
  inline __device__ void convert_from_float(uint4 &dst, Float8_ src) {  // NOLINT
    dst.x = float2_to_half2(src.x);
    dst.y = float2_to_half2(src.y);
    dst.z = float2_to_half2(src.z);
    dst.w = float2_to_half2(src.w);
  }
  
  inline __device__ void zero(uint16_t &dst) { dst = uint16_t(0); }  // NOLINT
  
  template <typename T>
  inline __device__ void zero(T &dst) {  // NOLINT
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
  
  template <typename T>
  __global__ void print_kernel(const T *tensor_data, int num) {
    printf("in kernel");
    for (int i = 0; i < num; ++i) {
      printf("%f ", static_cast<float>(tensor_data[i]));
    }
    printf("\n");
  }
  
  template <typename T, int Dh, int THREADS_PER_KEY, int THREADS_PER_VALUE,
            int THREADS_PER_BLOCK>
  __global__ void masked_multihead_attention_kernel(
      Masked_multihead_attention_params<T> params) {
    static_assert(Dh % THREADS_PER_KEY == 0, "");
    static_assert(Dh % THREADS_PER_VALUE == 0, "");
  
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
  
    extern __shared__ char smem_[];
  
    float *qk_smem = reinterpret_cast<float *>(smem_);
  
    char *logits_smem_ = smem_;
    // fp32 accum for logits
    float *logits_smem = reinterpret_cast<float *>(logits_smem_);
  
    T *out_smem = reinterpret_cast<T *>(smem_);
  
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];
    __shared__ T q_smem[Dh];
  
    const int bi = blockIdx.y;
    const int hi = blockIdx.x;
    const int bhi = bi * params.num_head + hi;
    const int tid = threadIdx.x;
  
    float qk_max = -FLT_MAX;
  
    // qkv [B, S=1, 3, num_head, head_dim]
    int qkv_base_offset = bi * 3 * params.num_head * Dh + hi * Dh;
  
    using Qk_vec = typename Qk_vec_<T, Dh>::Type;
    constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
    static_assert(Dh % QK_VEC_SIZE == 0 && Dh / QK_VEC_SIZE <= WARP_SIZE, "");
    constexpr int QK_VECS_PER_WARP = Dh / QK_VEC_SIZE;
  
    // cache_k, [B, num_head, head_dim / x, max_seq_len, x]
    // x == 4/8 for FP32/FP16, 128bit, 16Byte
    constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
    constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);
  
    const T *q_base = params.qkv;
    const T *k_base = params.qkv + params.num_head * Dh;
    const T *q_bias_base = params.qkv_bias;
    const T *k_bias_base = params.qkv_bias + params.num_head * Dh;
  
    if (tid < QK_VECS_PER_WARP) {
      int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;
      int qk_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
  
      Qk_vec q = *reinterpret_cast<const Qk_vec *>(&q_base[qk_offset]);
      Qk_vec k = *reinterpret_cast<const Qk_vec *>(&k_base[qk_offset]);
  
      Qk_vec q_bias =
          *reinterpret_cast<const Qk_vec *>(&q_bias_base[qk_bias_offset]);
      Qk_vec k_bias =
          *reinterpret_cast<const Qk_vec *>(&k_bias_base[qk_bias_offset]);
  
      q = add(q, q_bias);
      k = add(k, k_bias);
  
      *reinterpret_cast<Qk_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;
  
      int co = tid / QK_VECS_IN_16B;
      int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;
      int offset = bhi * params.max_seq_length * Dh +
                   co * params.max_seq_length * QK_ELTS_IN_16B +
                   params.timestep * QK_ELTS_IN_16B + ci;
      *reinterpret_cast<Qk_vec *>(&params.cache_kv[offset]) = k;
  
      float qk = dot<Qk_vec, Qk_vec>(q, k);
  #pragma unroll
      for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
      }
  
      qk *= params.inv_sqrt_dh;
      if (tid == 0) {
        // NOTE(wangxi): mask must be 0.0
        // T mask = params.attn_mask[
        //    bi * (params.timestep + 1) + params.timestep];
        // qk += static_cast<float>(mask);
        qk_max = qk;
        qk_smem[params.timestep] = qk;
      }
    }
    __syncthreads();
  
  #ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    if (bi == 0 && hi == 0 && tid == 0) {
      printf("=======q_out=======\n");
      for (int i = 0; i < Dh; ++i) printf("%f ", static_cast<float>(q_smem[i]));
      printf("\n");
    }
    __syncthreads();
  #endif
  
    using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
    constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
    static_assert(Dh % K_VEC_SIZE == 0, "");
    constexpr int K_ELTS_PER_THREAD = Dh / THREADS_PER_KEY;
    constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;
  
    int ko = tid / THREADS_PER_KEY;
    int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;
  
    K_vec q[K_VECS_PER_THREAD];
  #pragma unroll
    for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
      q[i] = *reinterpret_cast<const K_vec *>(
          &q_smem[ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
    }
  
    constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
    constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;
  
    T *k_cache = &params.cache_kv[bhi * params.max_seq_length * Dh + ki];
    int ti_end = div_up(params.timestep, K_PER_WARP) * K_PER_WARP;
  
    for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
      K_vec k[K_VECS_PER_THREAD];
  #pragma unroll
      for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
        int jj = ii * params.max_seq_length + ti;
        if (ti < params.timestep) {
          k[ii] = *reinterpret_cast<const K_vec *>(&k_cache[jj * QK_ELTS_IN_16B]);
        }
      }
  
      float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k) * params.inv_sqrt_dh;
  
      // bool is_mask = false;
      if (ti < params.timestep && tid % THREADS_PER_KEY == 0) {
        // qk_max = is_mask ? qk_max : fmaxf(qk_max, qk);
        T mask = params.attn_mask[bi * (params.timestep + 1) + ti];
        qk += static_cast<float>(mask);
        qk_max = fmaxf(qk_max, qk);
  
        qk_smem[ti] = qk;
      }
    }
  
  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
      qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
  
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
  
    if (lane == 0) {
      red_smem[warp] = qk_max;
    }
  
    __syncthreads();
  
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
  #pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
      qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
  
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);
  
  #ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    if (bi == 0 && hi == 0 && tid == 0) {
      printf("=======qk_out=======\n");
      for (int i = 0; i <= params.timestep; ++i) printf("%f ", qk_smem[i]);
      printf("qk_max=%f\n", qk_max);
    }
    __syncthreads();
  #endif
  
    float sum = 0.f;
    for (int ti = tid; ti <= params.timestep; ti += THREADS_PER_BLOCK) {
      // bool is_mask = false;
      // float logit = is_mask ? 0.f : __expf(qk_smem[ti] - qk_max);
      float logit = __expf(qk_smem[ti] - qk_max);
      sum += logit;
      qk_smem[ti] = logit;
    }
  
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);
  
    // FIXME(wangxi): need add 1.e-6f?
    float inv_sum = __fdividef(1.f, sum + 1.e-6f);
    for (int ti = tid; ti <= params.timestep; ti += THREADS_PER_BLOCK) {
      convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
    }
    __syncthreads();
  
    constexpr int V_VEC_SIZE = Dh / THREADS_PER_VALUE;
    using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;
  
    int vo = tid / THREADS_PER_VALUE;
    int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;
  
    T *v_cache = &params.cache_kv[params.batch_size * params.num_head *
                                      params.max_seq_length * Dh +
                                  bhi * params.max_seq_length * Dh + vi];
  
  #ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
  #else
    using V_vec_acum = V_vec;
  #endif
  
    V_vec_acum out;
    zero(out);
  
    constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    for (int ti = vo; ti < params.timestep; ti += V_PER_ITER) {
      V_vec v = *reinterpret_cast<const V_vec *>(&v_cache[ti * Dh]);
  #if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      float logit = logits_smem[ti];
      out = fma(logit, cast_to_float(v), out);
  #else
      T logit = logits_smem[ti];
      // Update the partial sums.
      out = fma(logit, v, out);
  #endif
    }
  
  #ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    if (bi == 0 && hi == 0 && tid == 0) {
      printf("======logits_out=====\n");
      for (int i = 0; i <= params.timestep; ++i) printf("%f ", logits_smem[i]);
      printf("\n");
    }
    __syncthreads();
  #endif
  
    if (vo == (params.timestep % V_PER_ITER)) {
      V_vec v = *reinterpret_cast<const V_vec *>(
          &params.qkv[2 * params.num_head * Dh + qkv_base_offset + vi]);
      V_vec v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[2 * params.num_head * Dh + hi * Dh + vi]);
      v = add(v, v_bias);
      *reinterpret_cast<V_vec *>(&v_cache[params.timestep * Dh]) = v;
  
  #if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      out = fma(logits_smem[params.timestep], cast_to_float(v), out);
  #else
      out = fma(logits_smem[params.timestep], v, out);
  #endif
    }
  
    __syncthreads();
  
  #pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {
      int midpoint = active_groups / 2;
  
      if (vo >= midpoint && vo < active_groups) {
  #ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(
            *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]),
            out);
  #else
        *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
  #endif
      }
      __syncthreads();
      if (vo < midpoint) {
        out = add(*reinterpret_cast<const V_vec *>(&out_smem[vo * Dh + vi]), out);
      }
      __syncthreads();
    }
  
    if (vo == 0) {
  #ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      convert_from_float(*reinterpret_cast<V_vec *>(&params.out[bhi * Dh + vi]),
                         out);
  #else
      *reinterpret_cast<V_vec *>(&params.out[bhi * Dh + vi]) = out;
  #endif
    }
  
  #ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
    __syncthreads();
    if (bi == 0 && hi == 0 && tid == 0) {
      printf("======fmha_out=====\n");
      for (int i = 0; i < Dh; ++i)
        printf("%f ", static_cast<float>(params.out[i]));
      printf("\n");
    }
  #endif
  }
  
  template <typename T>
  inline size_t smem_size_in_bytes(
      const Masked_multihead_attention_params<T> &params, int dim_head,
      int threads_per_value, int threads_per_block) {
    size_t qk_sz = div_up(params.timestep + 1, 4) * 16;
    size_t logits_sz = 0;
  
  #ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(T) != 4) {
      logits_sz = div_up(params.max_seq_length, 4) * 4 * sizeof(T);
    }
  #endif
    size_t softmax_sz = qk_sz + logits_sz;
  
    int rows_per_red = threads_per_block / threads_per_value;
    size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;
  
    return max(softmax_sz, red_sz);
  }
  
  #define MMHA_LAUNCH_KERNEL(T, Dh, THDS_PER_KEY, THDS_PER_VALUE,          \
                             THDS_PER_BLOCK, stream)                       \
    size_t smem_sz =                                                       \
        smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK); \
    dim3 grid(params.num_head, params.batch_size);                         \
    masked_multihead_attention_kernel<                                     \
        T, Dh, THDS_PER_KEY, THDS_PER_VALUE,                               \
        THDS_PER_BLOCK><<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)
  
  template <typename T, int Dh>
  void fmha_launch_kernel(const Masked_multihead_attention_params<T> &params,
                          const cudaStream_t &stream) {
    constexpr int THREADS_PER_VALUE = Dh * sizeof(T) / 16;
    if (params.timestep < 32) {
      MMHA_LAUNCH_KERNEL(T, Dh, 4, THREADS_PER_VALUE, 64, stream);
    } else if (params.timestep < 2048) {
      MMHA_LAUNCH_KERNEL(T, Dh, 2, THREADS_PER_VALUE, 128, stream);
    } else {
      MMHA_LAUNCH_KERNEL(T, Dh, 1, THREADS_PER_VALUE, 256, stream);
    }
  }
  
  template <typename T>
  void fmha(cudaStream_t stream, const T *qkv_tensor,
            const T *qkv_bias_tensor, const T *src_mask_tensor,
            T *cache_kv_tensor, T *out_tensor, int batch_size,
            int max_seq_length, int num_head, int dim_head, int timestep,
            float inv_sqrt_dh) {
    Masked_multihead_attention_params<T> params;
    params.out = out_tensor;
    params.qkv = qkv_tensor;
    params.qkv_bias = qkv_bias_tensor;
    params.attn_mask = src_mask_tensor;
    params.cache_kv = cache_kv_tensor;
  
    params.batch_size = batch_size;
    params.num_head = num_head;
    params.timestep = timestep;
    params.max_seq_length = max_seq_length;
    params.inv_sqrt_dh = inv_sqrt_dh;
  
    switch (dim_head) {
      case 32:
        fmha_launch_kernel<T, 32>(params, stream);
        break;
      case 64:
        fmha_launch_kernel<T, 64>(params, stream);
        break;
      case 128:
        fmha_launch_kernel<T, 128>(params, stream);
        break;
      default:
        assert(false);
    }
  }



template<typename T>
void TransformerDecoderPluginDynamic<T>::terminate() TRT_NOEXCEPT {
  if (p_gpu_bias_) {
    cudaFree(p_gpu_bias_);
  }
}

template<typename T>
int TransformerDecoderPluginDynamic<T>::initialize() TRT_NOEXCEPT {
  cudaMalloc(&p_gpu_bias_, sizeof(T) * bias_.size());
  cudaMemcpy(p_gpu_bias_, bias_.data(), bias_.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  return 0;
}

template<typename T>
TransformerDecoderPluginDynamic<T>::TransformerDecoderPluginDynamic(void const *serialData,
                                       size_t serialLength) {
  DeserializeValue(&serialData, &serialLength, &bias_);
  DeserializeValue(&serialData, &serialLength, &head_number_);
  DeserializeValue(&serialData, &serialLength, &head_size_);
  DeserializeValue(&serialData, &serialLength, &scale_);
  DeserializeValue(&serialData, &serialLength, &with_fp16_);
}

template<typename T>
size_t TransformerDecoderPluginDynamic<T>::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(bias_) + SerializedSize(head_number_)
         + SerializedSize(head_size_) + SerializedSize(scale_) + SerializedSize(with_fp16_); 
}

template<typename T>
void TransformerDecoderPluginDynamic<T>::serialize(void *buffer) const TRT_NOEXCEPT {
  SerializeValue(&buffer, bias_);
  SerializeValue(&buffer, head_number_);
  SerializeValue(&buffer, head_size_);
  SerializeValue(&buffer, scale_);
  SerializeValue(&buffer, with_fp16_);
}

template<typename T>
nvinfer1::DimsExprs TransformerDecoderPluginDynamic<T>::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs *inputs, int nb_inputs,
    nvinfer1::IExprBuilder &expr_builder) TRT_NOEXCEPT {
  // debug: (B, S, 3 * N * H, 1, 1)
  // input[0]: qkv_tensor,  [3, B, 1(seq_len), num_head * dim_head]
  // input[1]: bias_qk, [bsz, 1, 1, time_step(cache_seq_length)+1]
  // input[2]: kv_cache, [2, B, num_head, cache_seq_len, dim_head]
  // input[3]: gather_index, 
  // output[4]: [B, 1(seq_len), num_head * dim_head] 
  
  auto in_dims = inputs[0];
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  // ret.d[0] = in_dims.d[1];
  // ret.d[1] = in_dims.d[2];
  // ret.d[2] = in_dims.d[3];
  ret.d[0] = in_dims.d[0];
  ret.d[1] = in_dims.d[1];
  ret.d[2] = expr_builder.constant(head_number_ * head_size_);
  return ret;
}

template<typename T>
bool TransformerDecoderPluginDynamic<T>::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *in_out, int nb_inputs,
    int nb_outputs) TRT_NOEXCEPT {
  PADDLE_ENFORCE_NOT_NULL(
      in_out, platform::errors::InvalidArgument(
                  "The input of transformer decoder plugin shoule not be nullptr."));

  PADDLE_ENFORCE_LT(
      pos, nb_inputs + nb_outputs,
      platform::errors::InvalidArgument("The pos(%d) should be less than the "
                                        "num(%d) of the input and the output.",
                                        pos, nb_inputs + nb_outputs));
  (in_out && pos < (nb_inputs + nb_outputs));
  const nvinfer1::PluginTensorDesc &in = in_out[pos];
  if(pos == 0) {
    if(with_fp16_) {
      return (in_out[pos].type == nvinfer1::DataType::kHALF) &&
          (in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
    } else {
      return (in_out[pos].type == nvinfer1::DataType::kFLOAT) &&
          (in_out[pos].format == nvinfer1::PluginFormat::kLINEAR);
    }
  } else if (pos == 1 || pos == 2) {
    const nvinfer1::PluginTensorDesc &prev = in_out[pos - 1];
    return in.type == prev.type && in.format == prev.format;
  } else if(pos == 3) {
    return (in.type == nvinfer1::DataType::kINT32) &&
            (in.format == nvinfer1::PluginFormat::kLINEAR);
  } else { // output
    return (in.type == in_out[0].type) && (in.format == nvinfer1::PluginFormat::kLINEAR);
  }
}

template<typename T>
nvinfer1::DataType TransformerDecoderPluginDynamic<T>::getOutputDataType(
    int index, const nvinfer1::DataType *input_types,
    int nb_inputs) const TRT_NOEXCEPT {
  PADDLE_ENFORCE_EQ(index, 0, platform::errors::InvalidArgument(
                                  "The TransformerDecoder Plugin only has one input, so the "
                                  "index value should be 0, but get %d.",
                                  index));
  PADDLE_ENFORCE_EQ((input_types[0] == nvinfer1::DataType::kFLOAT)||(input_types[0] == nvinfer1::DataType::kHALF), true,
                    platform::errors::InvalidArgument(
                        "The input type should be half or float"));
  return input_types[0];
}


template<typename T>
int TransformerDecoderPluginDynamic<T>::enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                                const nvinfer1::PluginTensorDesc *output_desc,
                                const void *const *inputs, void *const *outputs,
                                void *workspace,
                                cudaStream_t stream) TRT_NOEXCEPT {
  return true;
  auto input_dims = input_desc[0].dims;
  auto input_type = input_desc[0].type;
  int bsz = input_dims.d[1];
  int max_seq_len = input_desc[2].dims.d[3];
  int time_step = input_desc[3].dims.d[0];
  VLOG(3) << "TransformerDecoderPluginDynamic::enqueue ----- ";
  VLOG(3) << "bsz: " << bsz << "; max_seq_len: " << max_seq_len << "; time_step: " << time_step;
  if (input_type == nvinfer1::DataType::kFLOAT) {
    VLOG(1) << "TRT Plugin DataType selected. TransformerDecoderPluginDynamic-->fp32";
    PADDLE_THROW(platform::errors::Fatal(
        "unsupported float format!!!"));

  } else if (input_type == nvinfer1::DataType::kHALF) {
    VLOG(1) << "TRT Plugin DataType selected. TransformerDecoderPluginDynamic-->fp16";
    const half *qkv_tensor = static_cast<const half *>(inputs[0]);
    const half *bias_qk = static_cast<const half *>(inputs[1]);
    const half *kv_cache = static_cast<const half *>(inputs[2]);
    const half *gather_index = static_cast<const half *>(inputs[3]);
    half *output = static_cast<half *>(outputs[0]);

    // void fmha(cudaStream_t stream, const T *qkv_tensor,
    //   const T *qkv_bias_tensor, const T *src_mask_tensor,
    //   T *cache_kv_tensor, T *out_tensor, int batch_size,
    //   int max_seq_length, int num_head, int dim_head, int timestep,
    //   float inv_sqrt_dh) {

    fmha<half>(stream, qkv_tensor, 
      const_cast<const half*>(p_gpu_bias_), bias_qk, 
      const_cast<half*>(kv_cache), output, bsz, 
      max_seq_len, head_number_, head_size_, time_step,
      scale_);
  }
  cudaDeviceSynchronize();
  return cudaGetLastError() != cudaSuccess;
}

template class TransformerDecoderPluginDynamic<half>; 

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
