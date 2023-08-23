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
    \brief Template for mmha kernel.
*/
#ifndef PADDLE_WITH_HIP
#pragma once

#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/mmha_util.cu.h"

namespace phi {
namespace fusion {

#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#define MMHA_USE_FP32_ACUM_FOR_OUT
#define MMHA_USE_FP32_ACUM_FOR_FMA

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

template <typename T>
struct Masked_multiquery_attention_params {
  // output buffer, [B, 1(seq_len), num_head * dim_head]
  T *out;
  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;
  // [bsz, seq_len]
  const int *cum_offsets;
  int mask_length;
  // whether to broadcast num_heads(2nd) dimension for attn_mask
  // in MMHA, if false, attn_mask shape should be
  // [bsz, num_heads, 1, time_step(cache_seq_length)+1]
  bool mask_broadcast_num_heads;

  // [2, B, num_head, max_seq_len(valid cache_seq_len), dim_head]
  // k [B, num_head, dim_head/x, max_seq_len, x], that is `seq_len` first
  // v [B, num_head, max_seq_len, dim_head]
  T *cache_kv;
  // [B, max_seq_len]
  const int *beam_cache_offset = nullptr;

  const int *sequence_lengths{nullptr};

  // The RoPE embedding, [2, B, rotary_seq_len, 1, dim_head]
  // rotary_emb_dims = 1 if pos_ids_extra is null else 2
  const float *rotary_emb;
  int rotary_emb_dims;
  int rotary_seq_len = 1;
  int num_head;
  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;
  int head_kv;
  bool split_kv;
  int timestep;  // cache_seq_length
  int max_seq_length;
  int seq_len;
  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;
  bool neox_rotary_style;
};

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

inline __device__ float4 hmma_fp32_tensorcore(const uint2 &a, uint32_t b) {
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
#if defined(MMQA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
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
#if defined(MMQA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
    return qk_hmma_dot_(q, k, inv_sqrt_dh);
#else
    return qk_dot_<4>(q, k, inv_sqrt_dh);
#endif
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

inline __device__ void convert_from_float(phi::float16 &dst,  // NOLINT
                                          float src) {
  dst = static_cast<phi::float16>(src);
}

inline __device__ void convert_from_float(uint4 &dst, Float8_ src) {  // NOLINT
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_BF16
inline __device__ void convert_from_float(__nv_bfloat16 &dst,  // NOLINT
                                          float src) {         // NOLINT
  dst = __float2bfloat16(src);
}

inline __device__ void convert_from_float(__nv_bfloat162 &dst,  // NOLINT
                                          float2 src) {         // NOLINT
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  dst = __float22bfloat162_rn(src);
#else
  dst = __floats2bfloat162_rn(src.x, src.y);
#endif
}

inline __device__ void convert_from_float(bf16_4_t &dst,  // NOLINT
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

inline __device__ void convert_from_float(bf16_4_t &dst,  // NOLINT
                                          float4 src) {   // NOLINT
  convert_from_float(
      dst, Float4_{make_float2(src.x, src.y), make_float2(src.z, src.w)});
}

inline __device__ void convert_from_float(bf16_8_t &dst,  // NOLINT
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

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc>
__global__ void masked_multiquery_attention_kernel(
    Masked_multiquery_attention_params<T> params,
    LoadFunc load_func,
    StoreFunc store_func) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int bi = blockIdx.y;
  if (params.sequence_lengths && params.sequence_lengths[bi] == 0) {
    return;
  }

  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0, "");

  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

  extern __shared__ char smem_[];

  float *qk_smem = reinterpret_cast<float *>(smem_);

  char *logits_smem_ = smem_;
  // fp32 accum for logits
  float *logits_smem = reinterpret_cast<float *>(logits_smem_);

  T *out_smem = reinterpret_cast<T *>(smem_);

  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  using Qk_vec_RoPE = typename Qk_vec_RoPE_<T, float, Dh_MAX>::Type;
  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  // beam id
  const int beami = bi % params.beam_width;
  // real batch id
  const int bbi = bi / params.beam_width;
  const int hi = blockIdx.x;

  const int hid = hi / (params.num_head / params.head_kv);
  const int bhi = bi * params.num_head + hi;
  const int bbhi = bbi * params.beam_width * params.num_head + hi;
  const int tid = threadIdx.x;
  const int bhi_kv = bi * params.head_kv + hid;
  const int bi_seq_len_offset = bi * params.max_seq_length;
  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;
  const int thi = params.cum_offsets ? ti * params.num_head + hi : -1;
  float qk_max = -FLT_MAX;
  float qk = 0;

  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];
  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

  // cache_k, [B, num_head, head_dim / x, max_seq_len, x]
  // x == 4/8 for FP32/FP16, 128bit, 16Byte
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec);

  // const T *q_base = params.qkv;
  // const T *k_base = params.qkv + params.num_head * Dh;

  if (tid < QK_VECS_PER_WARP) {
    Qk_vec q;
    zero(q);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(
          q, bi * params.num_head * Dh + hi * Dh + tid * QK_VEC_SIZE, 'q');
    }

    Qk_vec k;
    zero(k);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(
          k, bi * params.head_kv * Dh + hid * Dh + tid * QK_VEC_SIZE, 'k');
    }

    if (!params.neox_rotary_style) {
      if (params.rotary_emb_dims != 0) {
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.batch_size * Dh;
        Qk_vec_RoPE cos_emb, sin_emb;
        zero(cos_emb);
        zero(sin_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        apply_rotary_embedding(q, k, cos_emb, sin_emb);
      }
    } else {
      /* old rotary pos emb */
      if (params.rotary_emb_dims != 0) {
        int last_dim = Dh / params.rotary_emb_dims;
        int half_lastdim = last_dim / 2;
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.batch_size * Dh;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = tid / stride_all_lastdim * stride_all_lastdim +
                       (tid + stride) % (stride_all_lastdim);
        Qk_vec q_right;
        zero(q_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(
              q,
              bi * params.num_head * Dh + hi * Dh + right_id * QK_VEC_SIZE,
              'q');
        }
        Qk_vec k_right;
        zero(k_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(
              k_right,
              bi * params.head_kv * Dh + hid * Dh + right_id * QK_VEC_SIZE,
              'k');
        }
        Qk_vec_RoPE cos_emb;
        zero(cos_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;

        Qk_vec_RoPE sin_emb;
        zero(sin_emb);
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const Qk_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        float alpha = (tid % stride_all_lastdim) < stride
                          ? static_cast<float>(-1)
                          : static_cast<float>(1);
        q = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            q, q_right, cos_emb, sin_emb, alpha);
        k = apply_rotary_emb<Qk_vec, Qk_vec_RoPE>(
            k, k_right, cos_emb, sin_emb, alpha);
      }
    }

    *reinterpret_cast<Qk_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;

    int co = tid / QK_VECS_IN_16B;
    int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;
    int offset = bhi_kv * params.max_seq_length * Dh +
                 co * params.max_seq_length * QK_ELTS_IN_16B +
                 act_time_step * QK_ELTS_IN_16B + ci;
    if (blockIdx.x % (params.num_head / params.head_kv) == 0 &&
        (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B)) {
      *reinterpret_cast<Qk_vec *>(&params.cache_kv[offset]) = k;
    }

    qk = dot<Qk_vec, Qk_vec>(q, k);

    if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
      for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
      }
    }
  }
  if (QK_VECS_PER_WARP > WARP_SIZE) {
    constexpr int WARPS_PER_RED =
        (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
    qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
  }
  if (tid == 0) {
    // NOTE(wangxi): mask must be 0.0
    // T mask = params.attn_mask[
    //    bi * (params.timestep + 1) + params.timestep];
    // qk += static_cast<float>(mask);
    qk *= params.inv_sqrt_dh;
    qk_max = qk;
    qk_smem[act_time_step] = qk;
  }

  __syncthreads();

  using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  static_assert(Dh_MAX % K_VEC_SIZE == 0, "");
  constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

  int ko = tid / THREADS_PER_KEY;
  int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;

  static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD, "");

  K_vec q[K_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < K_VECS_PER_THREAD; ++i) {
    q[i] = *reinterpret_cast<const K_vec *>(
        &q_smem[ki + i * THREADS_PER_KEY * K_VEC_SIZE]);
  }

  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

  T *k_cache = &params.cache_kv[bhi_kv * params.max_seq_length * Dh + ki];
  T *k_cache_batch = &params.cache_kv[bhi_kv * params.max_seq_length * Dh + ki];
  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  const int *beam_offsets = params.beam_cache_offset
                                ? &params.beam_cache_offset[bi_seq_len_offset]
                                : nullptr;
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    const int beam_offset = beam_offsets ? beam_offsets[ti] * params.head_kv *
                                               params.max_seq_length * Dh
                                         : 0;
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + ti;
      if (ti < act_time_step) {
        if (beam_offset) {
          k[ii] =
              (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                  ? *reinterpret_cast<const K_vec *>(
                        &k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B])
                  : k_vec_zero;
        } else {
          k[ii] =
              (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                  ? *reinterpret_cast<const K_vec *>(
                        &k_cache[jj * QK_ELTS_IN_16B])
                  : k_vec_zero;
        }
      }
    }

    // NOTE(liyurui): We should multiple q with inv_sqrt_dh first, for dot(q, k)
    // may overflow with FP16 in large model.
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

    // bool is_mask = false;
    if (ti < act_time_step && tid % THREADS_PER_KEY == 0) {
      // qk_max = is_mask ? qk_max : fmaxf(qk_max, qk);
      auto mask_bhi = params.mask_broadcast_num_heads ? bi : bhi;
      // T mask = params.attn_mask[mask_bhi * (params.timestep + 1) + ti];
      if (params.attn_mask) {
        T mask = params.attn_mask[mask_bhi * params.mask_length + ti];
        qk += static_cast<float>(mask);
      }
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

  float sum = 0.f;
  for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
    // bool is_mask = false;
    // float logit = is_mask ? 0.f : __expf(qk_smem[ti] - qk_max);
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }

  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  // FIXME(wangxi): need add 1.e-6f?
  float inv_sum = __fdividef(1.f, sum + 1.e-6f);

  for (int ti = tid; ti <= act_time_step; ti += THREADS_PER_BLOCK) {
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }
  __syncthreads();
  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

  T *v_cache = &params.cache_kv[params.cache_batch_size * params.head_kv *
                                    params.max_seq_length * Dh +
                                bhi_kv * params.max_seq_length * Dh + vi];
  T *v_cache_batch = &params.cache_kv[params.batch_size * params.head_kv *
                                          params.max_seq_length * Dh +
                                      bhi_kv * params.max_seq_length * Dh + vi];

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif

  V_vec_acum out;
  zero(out);

  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if (Dh == Dh_MAX || vi < Dh) {
    for (int ti = vo; ti < act_time_step; ti += V_PER_ITER) {
      const int beam_offset = beam_offsets ? beam_offsets[ti] * params.head_kv *
                                                 params.max_seq_length * Dh
                                           : 0;
      V_vec v;
      if (beam_offset) {
        v = *reinterpret_cast<const V_vec *>(
            &v_cache_batch[beam_offset + ti * Dh]);
      } else {
        v = *reinterpret_cast<const V_vec *>(&v_cache[ti * Dh]);
      }
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      float logit = logits_smem[ti];
      out = fma(logit, cast_to_float(v), out);
#else
      DataType_ logit = static_cast<DataType_>(logits_smem[ti]);
      // Update the partial sums.
      out = fma(logit, v, out);
#endif
    }
  }

  if (vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v;
    load_func.template load<V_vec>(
        v, bi * params.head_kv * Dh + hid * Dh + vi, 'v');
    if (blockIdx.x % (params.num_head / params.head_kv) == 0)
      *reinterpret_cast<V_vec *>(&v_cache[act_time_step * Dh]) = v;

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    out = fma(logits_smem[act_time_step], cast_to_float(v), out);
#else
    out = fma(logits_smem[act_time_step], v, out);
#endif
  }

  __syncthreads();

  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2;
         active_groups /= 2) {
      int midpoint = active_groups / 2;

      if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        convert_from_float(
            *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]),
            out);
#else
        *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
      }
      __syncthreads();
      if (vo < midpoint && (Dh == Dh_MAX || vi < Dh)) {
        out =
            add(*reinterpret_cast<const V_vec *>(&out_smem[vo * Dh + vi]), out);
      }
      __syncthreads();
    }
  }

  if (vo == 0 && (Dh == Dh_MAX || vi < Dh)) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    V_vec tmp_out;
    convert_from_float(tmp_out, out);
    store_func.template store<V_vec>(tmp_out,
                                     thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#else
    store_func.template store<V_vec>(out,
                                     thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#endif
  }

#else
  assert(false);
#endif
}

template <typename T>
inline size_t smem_size_in_bytes(
    const Masked_multiquery_attention_params<T> &params,
    int dim_head,
    int threads_per_value,
    int threads_per_block) {
  size_t qk_sz = div_up(params.timestep + 1, 4) * 16;
  size_t logits_sz = 0;

#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS  // NOLINT
  if (sizeof(T) != 4) {
    logits_sz = div_up(params.max_seq_length, 4) * 4 * sizeof(T);
  }
#endif  // NOLINT
  size_t softmax_sz = qk_sz + logits_sz;

  int rows_per_red = threads_per_block / threads_per_value;
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;

  return max(softmax_sz, red_sz);
}

#define MMQA_LAUNCH_KERNEL(T,                                             \
                           Dh,                                            \
                           Dh_MAX,                                        \
                           THDS_PER_KEY,                                  \
                           THDS_PER_VALUE,                                \
                           THDS_PER_BLOCK,                                \
                           stream,                                        \
                           load_func,                                     \
                           store_func)                                    \
  size_t smem_sz =                                                        \
      smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);  \
  constexpr auto kernel_fn =                                              \
      masked_multiquery_attention_kernel<T,                               \
                                         Dh,                              \
                                         Dh_MAX,                          \
                                         THDS_PER_KEY,                    \
                                         THDS_PER_VALUE,                  \
                                         THDS_PER_BLOCK,                  \
                                         decltype(load_func),             \
                                         decltype(store_func)>;           \
  if (smem_sz > 0xc000) {                                                 \
    cudaFuncSetAttribute(                                                 \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz); \
  }                                                                       \
  dim3 grid(params.num_head, params.batch_size);                          \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                   \
      params, load_func, store_func)

template <typename T, int Dh, int Dh_MAX, typename LoadFunc, typename StoreFunc>
void fmqa_launch_kernel(const Masked_multiquery_attention_params<T> &params,
                        const cudaStream_t &stream,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  if (params.timestep < 32) {
    MMQA_LAUNCH_KERNEL(
        T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, stream, load_func, store_func);
  } else if (params.timestep < 2048) {
#if defined(MMQA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
    MMQA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       4,
                       THREADS_PER_VALUE,
                       256,
                       stream,
                       load_func,
                       store_func);
#else
    MMQA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       2,
                       THREADS_PER_VALUE,
                       128,
                       stream,
                       load_func,
                       store_func);
#endif
  } else {
    MMQA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       1,
                       THREADS_PER_VALUE,
                       256,
                       stream,
                       load_func,
                       store_func);
  }
}

template <typename T, typename LoadFunc, typename StoreFunc>
void fmqa_impl(const phi::GPUContext &dev_ctx,
               const Masked_multiquery_attention_params<T> &params,
               int dim_head,
               LoadFunc load_func,
               StoreFunc store_func) {
  switch (dim_head) {
    case 10:
      fmqa_launch_kernel<T, 10, 32>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 26:
      fmqa_launch_kernel<T, 26, 32>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 32:
      fmqa_launch_kernel<T, 32, 32>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 64:
      fmqa_launch_kernel<T, 64, 64>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 96:
      fmqa_launch_kernel<T, 96, 128>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 128:
      fmqa_launch_kernel<T, 128, 128>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    case 192:
      fmqa_launch_kernel<T, 192, 256>(
          params, dev_ctx.stream(), load_func, store_func);
      break;
    default:
      PADDLE_THROW(
          phi::errors::Unimplemented("Dim_head = %d is unsupport!", dim_head));
  }
}

template <typename T, typename StoreT = T, bool Smooth = false>
struct MMQAStore {
  explicit MMQAStore(StoreT *dst) : dst_(dst) {}

  template <typename Vec>
  __device__ void store(Vec &src, int idx) {
    *reinterpret_cast<Vec *>(dst_ + idx) = src;
  }

  StoreT *dst_;
};

template <typename T>
struct MMQAStore<T, T, true> {
  MMQAStore(T *dst, const T *shift, const T *smooth, const int cols)
      : dst_(dst), shift_(shift), smooth_(smooth), cols_(cols) {}

  template <typename Vec>
  __device__ void store(Vec &src, int idx) {
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using TVec = phi::AlignedVector<T, VecSize>;
    TVec src_vec;
    TVec shift_vec;
    TVec smooth_vec;

    *reinterpret_cast<Vec *>(&src_vec) = src;
    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);

#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src_vec[i] = (src_vec[i] + shift_vec[i]) * smooth_vec[i];
    }

    phi::Store<T, VecSize>(src_vec, dst_ + idx);
  }

  T *dst_;
  const T *shift_;
  const T *smooth_;
  const int cols_;
};

template <typename T>
struct MMQALoad {
  MMQALoad(const T *q, const T *k, const T *v) : q_(q), k_(k), v_(v) {}

  template <typename Vec>
  __device__ void load(Vec &dst, int idx, char load = false) {
    if (load == 'q')
      dst = *reinterpret_cast<const Vec *>(q_ + idx);
    else if ((load == 'k'))
      dst = *reinterpret_cast<const Vec *>(k_ + idx);
    else if ((load == 'v'))
      dst = *reinterpret_cast<const Vec *>(v_ + idx);
  }

  const T *q_;
  const T *k_;
  const T *v_;
};

template <typename T>
struct MMQAStore<T, int8_t> {
  MMQAStore(int8_t *dst,
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
  __device__ void store(Vec &src, int idx) {  // NOLINT
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using SrcVec = phi::AlignedVector<T, VecSize>;
    using DstVec = phi::AlignedVector<int8_t, VecSize>;

    SrcVec src_vec;
    *reinterpret_cast<Vec *>(&src_vec) = src;

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

  int8_t *dst_;
  const int quant_round_type_;
  const float quant_scale_;
  const float quant_max_bound_;
  const float quant_min_bound_;
};

template <typename T>
struct MMQAStore<T, int8_t, true> {
  MMQAStore(int8_t *dst,
            const T *shift,
            const T *smooth,
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
  __device__ void store(Vec &src, int idx) {  // NOLINT
    constexpr int VecSize = sizeof(Vec) / sizeof(T);
    using SrcVec = phi::AlignedVector<T, VecSize>;
    using DstVec = phi::AlignedVector<int8_t, VecSize>;

    SrcVec src_vec;
    DstVec dst_vec;
    SrcVec shift_vec;
    SrcVec smooth_vec;

    *reinterpret_cast<Vec *>(&src_vec) = src;
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

  int8_t *dst_;
  const T *shift_;
  const T *smooth_;
  const int cols_;
  const int quant_round_type_;
  const float quant_scale_;
  const float quant_max_bound_;
  const float quant_min_bound_;
};

template <typename T>
void DispatchFMQA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &query,
                  const phi::DenseTensor &key,
                  const phi::DenseTensor &value,
                  const Masked_multiquery_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f) {
  if (quant_fmha_out_scale > 0) {
    MMQALoad<T> load_func(query.data<T>(), key.data<T>(), value.data<T>());
    MMQAStore<T, int8_t> store_func(out_tensor->data<int8_t>(),
                                    quant_round_type,
                                    quant_fmha_out_scale,
                                    quant_max_bound,
                                    quant_min_bound);
    fmqa_impl(dev_ctx, params, dim_head, load_func, store_func);
  } else {
    MMQALoad<T> load_func(query.data<T>(), key.data<T>(), value.data<T>());
    MMQAStore<T> store_func(out_tensor->data<T>());
    fmqa_impl(dev_ctx, params, dim_head, load_func, store_func);
  }
}
template <typename T>
void DispatchFMQA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &query,
                  const phi::DenseTensor &key,
                  const phi::DenseTensor &value,
                  const phi::DenseTensor &shift,
                  const phi::DenseTensor &smooth,
                  const Masked_multiquery_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f) {
  if (quant_fmha_out_scale > 0) {
    MMQALoad<T> load_func(query.data<T>(), key.data<T>(), value.data<T>());
    MMQAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          num_head * dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    fmqa_impl(dev_ctx, params, dim_head, load_func, store_func);
  } else {
    MMQALoad<T> load_func(query.data<T>(), key.data<T>(), value.data<T>());
    MMQAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     num_head * dim_head);
    fmqa_impl(dev_ctx, params, dim_head, load_func, store_func);
  }
}

}  // namespace fusion
}  // namespace phi

#endif
