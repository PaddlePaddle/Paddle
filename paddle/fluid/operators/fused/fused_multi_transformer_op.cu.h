/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// This file has been adapted from FasterTransformer file:
// https://github.com/NVIDIA/FasterTransformer/blob/v4.0/fastertransformer/cuda/masked_multihead_attention.cu
// We add License in the head.

#include <cuda_fp16.h>
#include <float.h>

#include <cub/cub.cuh>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

// for debug
// #define _DEBUG_FUSED_MULTI_TRANSFORMER

template <typename T>
static void AllReduce(phi::DenseTensor &tensor,  // NOLINT
                      const int ring_id,
                      const int count,
                      const phi::GPUContext &ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto map = paddle::distributed::ProcessGroupMapFromGid::getInstance();

  if (map->has(ring_id)) {
    paddle::distributed::ProcessGroup *pg = map->get(ring_id);
    std::vector<phi::DenseTensor> in_tensor;
    std::vector<phi::DenseTensor> out_tensor;
    in_tensor.push_back(tensor);
    out_tensor.push_back(tensor);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::SUM;
    auto task = pg->AllReduce(in_tensor, out_tensor, opts);
    task->Wait();
  } else {
    auto dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(tensor.dtype()));
    int64_t numel = tensor.numel();
    const void *sendbuff = tensor.data<T>();
    auto place = ctx.GetPlace();
    void *recvbuff = tensor.mutable_data<T>(place);
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto stream = ctx.stream();
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, count, dtype, ncclSum, comm->comm(), stream));
  }
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

namespace {  // NOLINT

namespace plat = paddle::platform;
using float16 = plat::float16;

#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#define MMHA_USE_FP32_ACUM_FOR_OUT

template <typename T>
struct Masked_multihead_attention_params {
  // output buffer, [B, 1(seq_len), num_head * dim_head]
  T *out;
  // qkv_out, [B, 1(seq_len), 3, num_head * dim_head]
  const T *qkv;
  // bias, [3, num_head, dim_head]
  const T *qkv_bias;
  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;

  // [2, B, num_head, max_seq_len(valid cache_seq_len), dim_head]
  // k [B, num_head, dim_head/x, max_seq_len, x], that is `seq_len` first
  // v [B, num_head, max_seq_len, dim_head]
  T *cache_kv;

  int batch_size;
  int num_head;
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

// clang-format off

template <typename T, int Dh> struct Qk_vec_ {};
template <> struct Qk_vec_<float,    32> { using Type = float;    };
template <> struct Qk_vec_<float,    64> { using Type = float2;   };
template <> struct Qk_vec_<float,   128> { using Type = float4;   };
template <> struct Qk_vec_<float,   256> { using Type = float4;   };
template <> struct Qk_vec_<float16,  32> { using Type = uint32_t; };
template <> struct Qk_vec_<float16,  64> { using Type = uint32_t; };
template <> struct Qk_vec_<float16, 128> { using Type = uint2;    };
template <> struct Qk_vec_<float16, 256> { using Type = uint4;    };

template <typename T, int THREADS_PER_KEY> struct K_vec_ {};
template <> struct K_vec_<float,   4> { using Type = float;    };
template <> struct K_vec_<float,   2> { using Type = float2;   };
template <> struct K_vec_<float,   1> { using Type = float4;   };
template <> struct K_vec_<float16, 4> { using Type = uint32_t; };
template <> struct K_vec_<float16, 2> { using Type = uint2;    };
template <> struct K_vec_<float16, 1> { using Type = uint4;    };

template <typename T, int V_VEC_SIZE> struct V_vec_ {};
template <> struct V_vec_<float,   1> { using Type = float;    };
template <> struct V_vec_<float,   2> { using Type = float2;   };
template <> struct V_vec_<float,   4> { using Type = float4;   };
template <> struct V_vec_<float16, 2> { using Type = uint32_t; };
template <> struct V_vec_<float16, 4> { using Type = uint2;    };
template <> struct V_vec_<float16, 8> { using Type = uint4;    };

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template <typename T> struct V_vec_acum_fp32_ {};
// template <> struct V_vec_acum_fp32_<float>  { using Type = float;  };
// template <> struct V_vec_acum_fp32_<float2> { using Type = float2; };
template <> struct V_vec_acum_fp32_<float4> { using Type = float4; };
// template <> struct V_vec_acum_fp32_<uint32_t> { using Type = float2;   };
// template <> struct V_vec_acum_fp32_<uint2   > { using Type = Float4_;  };
template <> struct V_vec_acum_fp32_<uint4> { using Type = Float8_; };
#endif

// clang-format on

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
inline __device__ uint2 mul(uint2 a, float b) {
  uint2 res;
  res.x = mul<uint32_t, uint32_t, float>(a.x, b);
  res.y = mul<uint32_t, uint32_t, float>(a.y, b);
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
inline __device__ float2 mul(float2 a, float b) {
  float2 res;
  res.x = a.x * b;
  res.y = a.y * b;
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

inline __device__ void convert_from_float(plat::float16 &dst,  // NOLINT
                                          float src) {
  dst = static_cast<plat::float16>(src);
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

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK>
__global__ void masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params) {
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)

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
  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  const int bi = blockIdx.y;
  const int hi = blockIdx.x;
  const int bhi = bi * params.num_head + hi;
  const int tid = threadIdx.x;

  float qk_max = -FLT_MAX;
  float qk = 0;

  // qkv [B, S=1, 3, num_head, head_dim]
  int qkv_base_offset = bi * 3 * params.num_head * Dh + hi * Dh;

  constexpr int QK_VEC_SIZE = sizeof(Qk_vec) / sizeof(T);
  static_assert(Dh_MAX % QK_VEC_SIZE == 0, "");
  // Use block reduction if needed
  // static_assert(Dh_MAX / QK_VEC_SIZE <= WARP_SIZE, "");
  constexpr int QK_VECS_PER_WARP = Dh_MAX / QK_VEC_SIZE;

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

    Qk_vec q;
    zero(q);
    q = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
            ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_offset])
            : q;
    Qk_vec k;
    zero(k);
    k = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
            ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_offset])
            : k;

    Qk_vec q_bias;
    zero(q_bias);
    q_bias =
        (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
            ? *reinterpret_cast<const Qk_vec *>(&q_bias_base[qk_bias_offset])
            : q_bias;
    Qk_vec k_bias;
    zero(k_bias);
    k_bias =
        (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
            ? *reinterpret_cast<const Qk_vec *>(&k_bias_base[qk_bias_offset])
            : k_bias;

    q = add(q, q_bias);
    // TODO(wangxi): See this https://github.com/microsoft/unilm/issues/510
    //   we may not require k_bias.
    k = add(k, k_bias);

    *reinterpret_cast<Qk_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;

    int co = tid / QK_VECS_IN_16B;
    int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;
    int offset = bhi * params.max_seq_length * Dh +
                 co * params.max_seq_length * QK_ELTS_IN_16B +
                 params.timestep * QK_ELTS_IN_16B + ci;
    if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
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
    qk_smem[params.timestep] = qk;
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

  T *k_cache = &params.cache_kv[bhi * params.max_seq_length * Dh + ki];
  int ti_end = div_up(params.timestep, K_PER_WARP) * K_PER_WARP;

  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + ti;
      if (ti < params.timestep) {
        k[ii] =
            (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length)
                ? *reinterpret_cast<const K_vec *>(
                      &k_cache[jj * QK_ELTS_IN_16B])
                : k_vec_zero;
      }
    }

    // NOTE(liyurui): We should multiple q with inv_sqrt_dh first, for dot(q, k)
    // may overflow with FP16 in large model.
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

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

  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
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
  if (Dh == Dh_MAX || vi < Dh) {
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
  }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  if (bi == 0 && hi == 0 && tid == 0) {
    printf("======logits_out=====\n");
    for (int i = 0; i <= params.timestep; ++i) printf("%f ", logits_smem[i]);
    printf("\n");
  }
  __syncthreads();
#endif

  V_vec v_bias;
  zero(v_bias);
  if (vo == (params.timestep % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v = *reinterpret_cast<const V_vec *>(
        &params.qkv[2 * params.num_head * Dh + qkv_base_offset + vi]);
    v_bias = *reinterpret_cast<const V_vec *>(
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
#else
  assert(false);
#endif
}

template <typename T>
inline size_t smem_size_in_bytes(
    const Masked_multihead_attention_params<T> &params,
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

#define MMHA_LAUNCH_KERNEL(                                              \
    T, Dh, Dh_MAX, THDS_PER_KEY, THDS_PER_VALUE, THDS_PER_BLOCK, stream) \
  size_t smem_sz =                                                       \
      smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK); \
  dim3 grid(params.num_head, params.batch_size);                         \
  masked_multihead_attention_kernel<T,                                   \
                                    Dh,                                  \
                                    Dh_MAX,                              \
                                    THDS_PER_KEY,                        \
                                    THDS_PER_VALUE,                      \
                                    THDS_PER_BLOCK>                      \
      <<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params)

template <typename T, int Dh, int Dh_MAX>
void fmha_launch_kernel(const Masked_multihead_attention_params<T> &params,
                        const cudaStream_t &stream) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  if (params.timestep < 32) {
    MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, stream);
  } else if (params.timestep < 2048) {
    MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 2, THREADS_PER_VALUE, 128, stream);
  } else {
    MMHA_LAUNCH_KERNEL(T, Dh, Dh_MAX, 1, THREADS_PER_VALUE, 256, stream);
  }
}

template <typename T>
void fmha(const phi::GPUContext &dev_ctx,
          const Tensor &qkv_tensor,
          const Tensor &qkv_bias_tensor,
          const Tensor &src_mask_tensor,
          Tensor *cache_kv_tensor,
          Tensor *out_tensor,
          int batch_size,
          int max_seq_length,
          int num_head,
          int dim_head,
          int timestep,
          float inv_sqrt_dh) {
  Masked_multihead_attention_params<T> params;
  params.out = out_tensor->data<T>();
  params.qkv = qkv_tensor.data<T>();
  params.qkv_bias = qkv_bias_tensor.data<T>();
  params.attn_mask = src_mask_tensor.data<T>();
  params.cache_kv = cache_kv_tensor->data<T>();

  params.batch_size = batch_size;
  params.num_head = num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_length;
  params.inv_sqrt_dh = inv_sqrt_dh;

  switch (dim_head) {
    case 10:
      fmha_launch_kernel<T, 10, 32>(params, dev_ctx.stream());
      break;
    case 26:
      fmha_launch_kernel<T, 26, 32>(params, dev_ctx.stream());
      break;
    case 32:
      fmha_launch_kernel<T, 32, 32>(params, dev_ctx.stream());
      break;
    case 64:
      fmha_launch_kernel<T, 64, 64>(params, dev_ctx.stream());
      break;
    case 96:
      fmha_launch_kernel<T, 96, 128>(params, dev_ctx.stream());
      break;
    case 128:
      fmha_launch_kernel<T, 128, 128>(params, dev_ctx.stream());
      break;
    case 192:
      fmha_launch_kernel<T, 192, 256>(params, dev_ctx.stream());
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Dim_head = %d is unsupport!", dim_head));
  }
}

// NOTE: simd with 16Bytes(128bit), float is 4, float16 is 8
constexpr int VEC_16B = 16;

template <typename T>
__global__ void write_cache_k_kernel(T *cache_k,
                                     const T *k,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto k_src = reinterpret_cast<const uint4 *>(
      k + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  auto k_dst = reinterpret_cast<uint4 *>(
      cache_k + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // vec size
  int dim_head_div_x = dim_head / X_ELEMS;

  // FIXME(wangxi): num_head is not need?
  // if (out_idx >= num_head * dim_head_div_x * max_seq_len) return;
  if (out_idx >= dim_head_div_x * max_seq_len) return;

  int idx = out_idx;
  const int k_seq_len_id = idx % max_seq_len;
  // idx = (idx - k_seq_len_id) / max_seq_len;
  idx = idx / max_seq_len;
  const int k_vec_id = idx % dim_head_div_x;

  if (k_seq_len_id < seq_len) {
    k_dst[out_idx] = k_src[k_seq_len_id * dim_head_div_x + k_vec_id];
  }
}

template <typename T>
__global__ void write_cache_v_kernel(T *cache_v,
                                     const T *v,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int hi = blockIdx.z;

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto v_src = reinterpret_cast<const uint4 *>(
      v + bi * num_head * seq_len * dim_head + hi * seq_len * dim_head);
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<uint4 *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * seq_len) return;

  v_dst[idx] = v_src[idx];
}

template <typename T>
void write_cache_kv(const phi::GPUContext &dev_ctx,
                    T *cache_k,
                    T *cache_v,
                    const T *k,
                    const T *v,
                    const int bsz,
                    const int num_head,
                    const int seq_len,
                    const int max_seq_len,
                    const int dim_head) {
  constexpr int block_sz = 128;
  constexpr int x = VEC_16B / sizeof(T);

  assert(dim_head % x == 0);
  PADDLE_ENFORCE_EQ(
      dim_head % x,
      0,
      platform::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, x));

  int max_size = max_seq_len * dim_head / x;
  int size = seq_len * dim_head / x;
  dim3 grid(div_up(max_size, block_sz), bsz, num_head);
  dim3 grid_v(div_up(size, block_sz), bsz, num_head);

  // transpose [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  write_cache_k_kernel<<<grid, block_sz, 0, dev_ctx.stream()>>>(
      cache_k, k, num_head, dim_head, seq_len, max_seq_len);

  // copy [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  write_cache_v_kernel<<<grid_v, block_sz, 0, dev_ctx.stream()>>>(
      cache_v, v, num_head, dim_head, seq_len, max_seq_len);
}

}  // namespace

}  // namespace operators
}  // namespace paddle
