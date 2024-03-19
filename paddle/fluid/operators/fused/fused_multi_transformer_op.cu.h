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

#pragma once

#include <fstream>
#include <iomanip>
#include "paddle/phi/kernels/flash_attn_kernel.h"

#include "paddle/fluid/operators/fused/mmha_util.cu.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

DECLARE_string(fmha_mode);
DECLARE_int64(custom_allreduce_one_shot_threshold);
DECLARE_int64(custom_allreduce_two_shot_threshold);

namespace paddle {
namespace operators {

inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

static float CPUHalfConvert2Float(const uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  // const float exp_scale = 0x1.0p-112f;
  constexpr uint32_t scale_bits = (uint32_t)15 << 23;
  float exp_scale_val;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

template <typename T>
static void PrintMatrix(const T *mat_d, int num, std::string name) {
  // if (FLAGS_cublaslt_exhaustive_search_times != 114514) return;

  std::vector<T> tmp(num);
  cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);

  std::ofstream outfile;
  outfile.open(name + ".txt", std::ios::out);
  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    if (std::is_same<T, int8_t>::value) {
      ss << static_cast<int>(tmp[i]) << std::endl;
    } else {
      ss << std::setprecision(8) << (float)(tmp[i]) << std::endl;  // NOLINT
    }
  }
  outfile << ss.str();
  outfile.close();
}

static void PrintHalfMatrix(const void *mat_d_ptr, int num, std::string name) {
  VLOG(0) << "PRINT HALF MATRIX Num is: " << num;
  const uint16_t *mat_d = reinterpret_cast<const uint16_t *>(mat_d_ptr);
  std::vector<uint16_t> tmp(num);
  cudaMemcpy(tmp.data(), mat_d, sizeof(uint16_t) * num, cudaMemcpyDeviceToHost);

  std::ofstream outfile;
  outfile.open(name + ".txt", std::ios::out);
  std::stringstream ss;

  for (int i = 0; i < num; ++i) {
    ss << std::setprecision(8) << CPUHalfConvert2Float(tmp[i]) << std::endl;
  }
  outfile << ss.str();
  outfile.close();
}

template <typename T>
struct Load {
  explicit Load(const T *src) : src_(src) {}

  template <int VecSize>
  __device__ void load(phi::AlignedVector<T, VecSize> *dst, int idx) {
    phi::Load<T, VecSize>(src_ + idx, dst);
  }

  const T *src_;
};

template <typename T, bool Smooth = false>
struct Store {
  explicit Store(T *dst) : dst_(dst) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src, int idx) {
    phi::Store<T, VecSize>(src, dst_ + idx);
  }

  T *dst_;
};

template <typename T>
struct Store<T, true> {
  Store(T *dst, const T *shift, const T *smooth, const int cols)
      : dst_(dst), shift_(shift), smooth_(smooth), cols_(cols) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src, int idx) {
    using Vec = phi::AlignedVector<T, VecSize>;
    Vec shift_vec;
    Vec smooth_vec;

    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src[i] = (src[i] + shift_vec[i]) * smooth_vec[i];
    }
    phi::Store<T, VecSize>(src, dst_ + idx);
  }

  T *dst_;
  const T *shift_;
  const T *smooth_;
  const int cols_;
};

template <typename T>
struct DequantLoad {
  DequantLoad(const int32_t *src, const float *dequant_scales, const int cols)
      : src_(src), dequant_scales_(dequant_scales), cols_(cols) {}

  template <int VecSize>
  __device__ void load(phi::AlignedVector<T, VecSize> *dst, int idx) {
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
    *dst = dst_vec;
  }

  const int32_t *src_;
  const float *dequant_scales_;
  const int cols_;
};

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
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(rint(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <typename T, bool Smooth = false>
struct QuantStore {
  QuantStore(int8_t *dst,
             const int quant_round_type,
             const float quant_scale,
             const float quant_max_bound,
             const float quant_min_bound)
      : dst_(dst),
        quant_round_type_(quant_round_type),
        quant_scale_(quant_scale),
        quant_max_bound_(quant_max_bound),
        quant_min_bound_(quant_min_bound) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src,  // NOLINT
                        int idx) {                            // NOLINT
    using DstVec = phi::AlignedVector<int8_t, VecSize>;

    DstVec dst_vec;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      dst_vec[i] = QuantHelperFunc<float, int8_t>(static_cast<float>(src[i]),
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
struct QuantStore<T, true> {
  QuantStore(int8_t *dst,
             const T *shift,
             const T *smooth,
             const int cols,
             const int quant_round_type,
             const float quant_scale,
             const float quant_max_bound,
             const float quant_min_bound)
      : dst_(dst),
        shift_(shift),
        smooth_(smooth),
        cols_(cols),
        quant_round_type_(quant_round_type),
        quant_scale_(quant_scale),
        quant_max_bound_(quant_max_bound),
        quant_min_bound_(quant_min_bound) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src,  // NOLINT
                        int idx) {                            // NOLINT
    using DstVec = phi::AlignedVector<int8_t, VecSize>;
    using Vec = phi::AlignedVector<T, VecSize>;

    DstVec dst_vec;
    Vec shift_vec;
    Vec smooth_vec;

    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src[i] = (src[i] + shift_vec[i]) * smooth_vec[i];
      dst_vec[i] = QuantHelperFunc<float, int8_t>(static_cast<float>(src[i]),
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
  const T *shift_;
  const T *smooth_;
  const int cols_;
};

template <typename T, typename LoadT = T>
struct MMHALoad {
  explicit MMHALoad(const LoadT *src) : src_(src) {}

  template <typename Vec>
  __device__ void load(Vec &dst, int idx) {
    dst = *reinterpret_cast<const Vec *>(src_ + idx);
  }

  const LoadT *src_;
};

template <typename T, typename StoreT = T, bool Smooth = false>
struct MMHAStore {
  explicit MMHAStore(StoreT *dst) : dst_(dst) {}

  template <typename Vec>
  __device__ void store(Vec &src, int idx) {
    *reinterpret_cast<Vec *>(dst_ + idx) = src;
  }

  StoreT *dst_;
};

template <typename T>
struct MMHAStore<T, T, true> {
  MMHAStore(T *dst, const T *shift, const T *smooth, const int cols)
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
struct MMHALoad<T, int32_t> {
  MMHALoad(const int32_t *src, const float *dequant_scales, const int cols)
      : src_(src), dequant_scales_(dequant_scales), cols_(cols) {}

  template <typename Vec>
  __device__ void load(Vec &dst, int idx) {
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
    dst = *reinterpret_cast<Vec *>(&dst_vec);
  }

  const int32_t *src_;
  const float *dequant_scales_;
  const int cols_;
};

template <typename T>
struct MMHAStore<T, int8_t> {
  MMHAStore(int8_t *dst,
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
struct MMHAStore<T, int8_t, true> {
  MMHAStore(int8_t *dst,
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
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char *, float *>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

template <typename T>
struct CudaSwishFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta = 1.0;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  // swish(x) = x / (1 + exp(-beta * x))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    return static_cast<T>(x / (one + exp(-b * x)));
  }
};

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
    // std::vector<phi::DenseTensor> in_tensor;
    // std::vector<phi::DenseTensor> out_tensor;
    // in_tensor.push_back(tensor);
    // out_tensor.push_back(tensor);
    paddle::distributed::AllreduceOptions opts;
    opts.reduce_op = distributed::ReduceOp::SUM;
    auto task = pg->AllReduce(&tensor, tensor, opts, false, true);
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

#define MMHA_USE_FP32_ACUM_FOR_LOGITS
#define MMHA_USE_FP32_ACUM_FOR_OUT
#define MMHA_USE_FP32_ACUM_FOR_FMA
// #define MMHA_USE_HMMA_FOR_REDUCTION

template <typename T>
struct Masked_multihead_attention_params {
  // output buffer, [B, 1(seq_len), num_head * dim_head]
  T *out;
  // qkv_out, [B, 1(seq_len), 3, num_head * dim_head]
  const T *qkv;
  // bias, [3, num_head, dim_head]
  T *qkv_bias;
  // [bsz, seq_len]
  const int *cum_offsets;
  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;
  int mask_length;
  // whether to broadcast num_heads(2nd) dimension for attn_mask
  // in MMHA, if false, attn_mask shape should be
  // [bsz, num_heads, 1, time_step(cache_seq_length)+1]
  bool mask_broadcast_num_heads;

  // [2, B, num_head, max_seq_len(valid cache_seq_len), dim_head]
  // k [B, num_head, dim_head/x, max_seq_len, x], that is `seq_len` first
  // v [B, num_head, max_seq_len, dim_head]
  T *cache_kv = nullptr;
  // [B, max_seq_len]
  const int *beam_cache_offset = nullptr;

  const int *sequence_lengths{nullptr};

  // The RoPE embedding, [2, B, rotary_seq_len, 1, dim_head]
  // rotary_emb_dims = 1 if pos_ids_extra is null else 2
  const float *rotary_emb;
  int rotary_bsz;
  int rotary_emb_dims;
  int rotary_seq_len = 1;

  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;
  int num_head;
  int timestep;  // cache_seq_length
  int seq_len;
  int max_seq_length;

  int gqa_group_size;
  int gqa_num_per_partitions;

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;

  bool add_qkv_bias;
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
__global__ __launch_bounds__(
    THREADS_PER_BLOCK) void masked_multihead_attention_kernel_int8(  // NOLINT
    Masked_multihead_attention_params<T> params,
    LoadFunc load_func,
    StoreFunc store_func,
    uint8_t *cache_kv_I,
    float cache_k_quant_scale,
    float cache_v_quant_scale,
    float cache_k_dequant_scale,
    float cache_v_dequant_scale) {
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
  using QK_Packed_Int8_t =
      typename packed_type<uint8_t, num_elems<Qk_vec>::value>::type;
  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  // beam id
  const int beami = bi % params.beam_width;
  // real batch id
  const int bbi = bi / params.beam_width;
  const int hi = blockIdx.x;
  const int bhi = bi * params.num_head + hi;
  const int bbhi = bbi * params.beam_width * params.num_head + hi;
  const int tid = threadIdx.x;

  const int bi_seq_len_offset = bi * params.max_seq_length;

  float qk_max = -FLT_MAX;
  float qk = 0;

  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];

  __shared__ float k_q_scale;
  __shared__ float k_dq_scale;
  __shared__ float v_q_scale;
  __shared__ float v_dq_scale;

  k_q_scale = cache_k_quant_scale;
  k_dq_scale = cache_k_dequant_scale;
  v_q_scale = cache_v_quant_scale;
  v_dq_scale = cache_v_dequant_scale;

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

  // const T *q_base = params.qkv;
  // const T *k_base = params.qkv + params.num_head * Dh;
  T *q_bias_base = nullptr;
  T *k_bias_base = nullptr;

  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + params.num_head * Dh;
  }

  if (tid < QK_VECS_PER_WARP) {
    int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    int qk_bias_offset = hi * Dh + tid * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    // q = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
    //         ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_offset])
    //         : q;
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset);
    }

    Qk_vec k;
    zero(k);
    // k = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
    //         ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_offset])
    //         : k;
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(k, params.num_head * Dh + qk_offset);
    }

    if (params.add_qkv_bias) {
      Qk_vec q_bias;
      zero(q_bias);
      Qk_vec k_bias;
      zero(k_bias);

      q_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&q_bias_base[qk_bias_offset])
              : q_bias;
      k_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&k_bias_base[qk_bias_offset])
              : k_bias;

      q = add(q, q_bias);
      // TODO(wangxi): See this https://github.com/microsoft/unilm/issues/510
      //   we may not require k_bias.
      k = add(k, k_bias);
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
        int qk_right_offset = qkv_base_offset + right_id * QK_VEC_SIZE;
        int qk_right_bias_offset = hi * Dh + right_id * QK_VEC_SIZE;
        Qk_vec q_right;
        zero(q_right);
        // q_right =
        //     (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //         ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_right_offset])
        //         : q_right;
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(q_right, qk_right_offset);
        }
        Qk_vec k_right;
        zero(k_right);
        // k_right =
        //     (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //         ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_right_offset])
        //         : k_right;
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(
              k_right, params.num_head * Dh + qk_right_offset);
        }

        if (params.add_qkv_bias) {
          Qk_vec q_right_bias;
          zero(q_right_bias);
          q_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
                             ? *reinterpret_cast<const Qk_vec *>(
                                   &q_bias_base[qk_right_bias_offset])
                             : q_right_bias;
          Qk_vec k_right_bias;
          zero(k_right_bias);
          k_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
                             ? *reinterpret_cast<const Qk_vec *>(
                                   &k_bias_base[qk_right_bias_offset])
                             : k_right_bias;

          q_right = add(q_right, q_right_bias);
          k_right = add(k_right, k_right_bias);
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
    int offset = bhi * params.max_seq_length * Dh +
                 co * params.max_seq_length * QK_ELTS_IN_16B +
                 act_time_step * QK_ELTS_IN_16B + ci;
    // quant k and store the int8 value into cache kv
    if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
      QK_Packed_Int8_t k_tmp = round_tmp<QK_Packed_Int8_t, Qk_vec>(
          mul<Qk_vec, float, Qk_vec>(k_q_scale, k));
      *reinterpret_cast<QK_Packed_Int8_t *>(&cache_kv_I[offset]) = k_tmp;
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
  using K_vec_I = typename K_vec_I_<T, THREADS_PER_KEY>::Type;
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

  uint8_t *k_cache_I = &cache_kv_I[bhi * params.max_seq_length * Dh + ki];
  uint8_t *k_cache_batch_I =
      &cache_kv_I[bbhi * params.max_seq_length * Dh + ki];

  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  const int *beam_offsets = params.beam_cache_offset
                                ? &params.beam_cache_offset[bi_seq_len_offset]
                                : nullptr;
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    const int beam_offset = beam_offsets ? beam_offsets[ti] * params.num_head *
                                               params.max_seq_length * Dh
                                         : 0;
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + ti;
      // get k from the cache_kv, and dequant k for qk operation
      if (ti < act_time_step) {
        if (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * params.max_seq_length) {
          mul_pointer_v2<K_vec>(
              &k[ii],
              k_dq_scale,
              // (beam_offset) ? reinterpret_cast<K_vec_I* >(k_cache_batch_I +
              // beam_offset + jj * QK_ELTS_IN_16B) : reinterpret_cast<K_vec_I*
              // >(k_cache_I + jj * QK_ELTS_IN_16B));
              reinterpret_cast<K_vec_I *>(k_cache_I + jj * QK_ELTS_IN_16B));
        } else {
          k[ii] = k_vec_zero;
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
      T mask = params.attn_mask[mask_bhi * params.mask_length + ti];
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
  using V_Packed_Int8_t =
      typename packed_type<uint8_t, num_elems<V_vec>::value>::type;

  int vo = tid / THREADS_PER_VALUE;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;

  uint8_t *v_cache_I = &cache_kv_I[params.cache_batch_size * params.num_head *
                                       params.max_seq_length * Dh +
                                   bhi * params.max_seq_length * Dh + vi];
  uint8_t *v_cache_batch_I =
      &cache_kv_I[params.batch_size * params.num_head * params.max_seq_length *
                      Dh +
                  bbhi * params.max_seq_length * Dh + vi];

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
      const int beam_offset =
          beam_offsets
              ? beam_offsets[ti] * params.num_head * params.max_seq_length * Dh
              : 0;
      V_vec v;
      mul_pointer_v2<V_vec>(
          &v,
          v_dq_scale,
          // (beam_offset) ?  reinterpret_cast<V_Packed_Int8_t*
          // >(v_cache_batch_I + beam_offset + ti * Dh) :
          // reinterpret_cast<V_Packed_Int8_t* >(v_cache_I + ti * Dh));
          reinterpret_cast<V_Packed_Int8_t *>(v_cache_I + ti * Dh));
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

  V_vec v_bias;
  zero(v_bias);
  if (vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    // V_vec v = *reinterpret_cast<const V_vec *>(
    //     &params.qkv[2 * params.num_head * Dh + qkv_base_offset + vi]);
    V_vec v;
    load_func.template load<V_vec>(
        v, 2 * params.num_head * Dh + qkv_base_offset + vi);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[2 * params.num_head * Dh + hi * Dh + vi]);
      v = add(v, v_bias);
    }

    V_Packed_Int8_t v_tmp = round_tmp<V_Packed_Int8_t, V_vec>(
        mul<V_vec, float, V_vec>(v_q_scale, v));
    *reinterpret_cast<V_Packed_Int8_t *>(&v_cache_I[act_time_step * Dh]) =
        v_tmp;

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
    // convert_from_float(*reinterpret_cast<V_vec *>(&params.out[bhi * Dh +
    // vi]),
    //                    out);
    V_vec tmp_out;
    convert_from_float(tmp_out, out);
    store_func.template store<V_vec>(tmp_out, bhi * Dh + vi);
#else
    // *reinterpret_cast<V_vec *>(&params.out[bhi * Dh + vi]) = out;
    store_func.template store<V_vec>(out, bhi * Dh + vi);
#endif
  }

#else
  assert(false);
#endif
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc>
__global__ void masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params,
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
  const int kv_hi = hi / params.gqa_num_per_partitions;
  const int bhi = bi * params.num_head + hi;
  const int bbhi = bbi * params.beam_width * params.num_head + hi;
  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;
  const int thi = params.cum_offsets ? ti * params.num_head + hi : -1;
  const int tid = threadIdx.x;

  const int bi_seq_len_offset = bi * params.max_seq_length;

  float qk_max = -FLT_MAX;
  float qk = 0;

  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];

  // qkv [B, S=1, num_head + 2 * gqa_group_size, head_dim]
  int qkv_base_offset = bi * (params.num_head + 2 * params.gqa_group_size) * Dh;

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
  T *q_bias_base = nullptr;
  T *k_bias_base = nullptr;

  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + params.num_head * Dh;
  }

  if (tid < QK_VECS_PER_WARP) {
    int qk_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    const int q_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
    const int k_bias_offset = kv_hi * Dh + tid * QK_VEC_SIZE;

    Qk_vec q;
    zero(q);
    // q = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
    //         ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_offset])
    //         : q;
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(q, qk_offset + hi * Dh);
    }

    Qk_vec k;
    zero(k);
    // k = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
    //         ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_offset])
    //         : k;
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Qk_vec>(
          k, params.num_head * Dh + qk_offset + kv_hi * Dh);
    }

    if (params.add_qkv_bias) {
      Qk_vec q_bias;
      zero(q_bias);
      Qk_vec k_bias;
      zero(k_bias);

      q_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&q_bias_base[q_bias_offset])
              : q_bias;
      k_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const Qk_vec *>(&k_bias_base[k_bias_offset])
              : k_bias;

      q = add(q, q_bias);
      // TODO(wangxi): See this https://github.com/microsoft/unilm/issues/510
      //   we may not require k_bias.
      k = add(k, k_bias);
    }

    if (!params.neox_rotary_style) {
      if (params.rotary_emb_dims != 0) {
        int rotary_offset = bi * Dh + tid * QK_VEC_SIZE;
        const float *cos_base = params.rotary_emb;
        const float *sin_base = params.rotary_emb + params.rotary_bsz * Dh;
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
        const float *sin_base = params.rotary_emb + params.rotary_bsz * Dh;
        int stride = half_lastdim / QK_VEC_SIZE;
        int stride_all_lastdim = 2 * stride;
        int right_id = tid / stride_all_lastdim * stride_all_lastdim +
                       (tid + stride) % (stride_all_lastdim);
        int q_right_offset = qkv_base_offset + hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_offset = qkv_base_offset + params.num_head * Dh +
                             kv_hi * Dh + right_id * QK_VEC_SIZE;
        Qk_vec q_right;
        zero(q_right);
        // q_right =
        //     (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //         ? *reinterpret_cast<const Qk_vec *>(&q_base[qk_right_offset])
        //         : q_right;
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(q_right, q_right_offset);
        }
        Qk_vec k_right;
        zero(k_right);
        // k_right =
        //     (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //         ? *reinterpret_cast<const Qk_vec *>(&k_base[qk_right_offset])
        //         : k_right;
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<Qk_vec>(k_right, k_right_offset);
        }

        // if (params.add_qkv_bias) {
        //   Qk_vec q_right_bias;
        //   zero(q_right_bias);
        //   q_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //                      ? *reinterpret_cast<const Qk_vec *>(
        //                            &q_bias_base[qk_right_bias_offset])
        //                      : q_right_bias;
        //   Qk_vec k_right_bias;
        //   zero(k_right_bias);
        //   k_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
        //                      ? *reinterpret_cast<const Qk_vec *>(
        //                            &k_bias_base[qk_right_bias_offset])
        //                      : k_right_bias;

        //   q_right = add(q_right, q_right_bias);
        //   k_right = add(k_right, k_right_bias);
        // }

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
    // int offset = bhi * params.max_seq_length * Dh +
    //              co * params.max_seq_length * QK_ELTS_IN_16B +
    //              act_time_step * QK_ELTS_IN_16B + ci;

    int offset = bi * params.gqa_group_size * params.max_seq_length * Dh +
                 kv_hi * params.max_seq_length * Dh +
                 co * params.max_seq_length * QK_ELTS_IN_16B +
                 act_time_step * QK_ELTS_IN_16B + ci;

    // quant k and store the int8 value into cache kv
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
    qk *= params.inv_sqrt_dh;
    qk_max = qk;
    qk_smem[act_time_step] = qk;
  }
  __syncthreads();

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("=======q_out=======\n");
  //   for (int i = 0; i < Dh; ++i) printf("%f ",
  //   static_cast<float>(q_smem[i])); printf("\n");
  // }
  // __syncthreads();
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

  T *k_cache =
      &params.cache_kv[bi * params.gqa_group_size * params.max_seq_length * Dh +
                       kv_hi * params.max_seq_length * Dh + ki];
  // T *k_cache_batch = &params.cache_kv[bbhi * params.max_seq_length * Dh +
  // ki];

  int ti_end = div_up(act_time_step, K_PER_WARP) * K_PER_WARP;

  const int *beam_offsets = params.beam_cache_offset
                                ? &params.beam_cache_offset[bi_seq_len_offset]
                                : nullptr;
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    const int beam_offset = beam_offsets ? beam_offsets[ti] * params.num_head *
                                               params.max_seq_length * Dh
                                         : 0;
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
      int jj = ii * params.max_seq_length + ti;
      // get k from the cache_kv, and dequant k for qk operation
      if (ti < act_time_step) {
        if (beam_offset) {
          // k[ii] =
          //     (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh *
          //     params.max_seq_length)
          //         ? *reinterpret_cast<const K_vec *>(
          //               &k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B])
          //         : k_vec_zero;
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

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("=======qk_out=======\n");
  //   for (int i = 0; i <= params.timestep; ++i) printf("%f ", qk_smem[i]);
  //   printf("qk_max=%f\n", qk_max);
  // }
  // __syncthreads();
#endif

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

  T *v_cache =
      &params.cache_kv[params.cache_batch_size * params.gqa_group_size *
                           params.max_seq_length * Dh +
                       bi * params.gqa_group_size * params.max_seq_length * Dh +
                       kv_hi * params.max_seq_length * Dh + vi];
  // T *v_cache_batch = &params.cache_kv[params.batch_size * params.num_head *
  //                                         params.max_seq_length * Dh +
  //                                     bbhi * params.max_seq_length * Dh +
  //                                     vi];

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
      const int beam_offset =
          beam_offsets
              ? beam_offsets[ti] * params.num_head * params.max_seq_length * Dh
              : 0;
      V_vec v;
      if (beam_offset) {
        // v = *reinterpret_cast<const V_vec *>(
        //     &v_cache_batch[beam_offset + ti * Dh]);
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

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("======logits_out=====\n");
  //   for (int i = 0; i <= params.timestep; ++i) printf("%f ", logits_smem[i]);
  //   printf("\n");
  // }
  // __syncthreads();
#endif

  V_vec v_bias;
  zero(v_bias);
  if (vo == (act_time_step % V_PER_ITER) && (Dh == Dh_MAX || vi < Dh)) {
    // V_vec v = *reinterpret_cast<const V_vec *>(
    //     &params.qkv[2 * params.num_head * Dh + qkv_base_offset + vi]);
    V_vec v;
    load_func.template load<V_vec>(v,
                                   params.num_head * Dh +
                                       params.gqa_group_size * Dh +
                                       qkv_base_offset + kv_hi * Dh + vi);
    if (params.add_qkv_bias) {
      v_bias = *reinterpret_cast<const V_vec *>(
          &params.qkv_bias[(params.num_head + params.gqa_group_size) * Dh +
                           kv_hi * Dh + vi]);
      v = add(v, v_bias);
    }

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
    // convert_from_float(*reinterpret_cast<V_vec *>(&params.out[bhi * Dh +
    // vi]),
    //                    out);
    V_vec tmp_out;
    convert_from_float(tmp_out, out);
    store_func.template store<V_vec>(tmp_out,
                                     thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#else
    // *reinterpret_cast<V_vec *>(&params.out[bhi * Dh + vi]) = out;
    store_func.template store<V_vec>(out,
                                     thi != -1 ? thi * Dh + vi : bhi * Dh + vi);
#endif
  }

#ifdef _DEBUG_FUSED_MULTI_TRANSFORMER
  // __syncthreads();
  // if (bi == 0 && hi == 0 && tid == 0) {
  //   printf("======fmha_out=====\n");
  //   for (int i = 0; i < Dh; ++i)
  //     printf("%f ", static_cast<float>(params.out[i]));
  //   printf("\n");
  // }
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

#define MMHA_LAUNCH_KERNEL_INT8(T,                                            \
                                Dh,                                           \
                                Dh_MAX,                                       \
                                THDS_PER_KEY,                                 \
                                THDS_PER_VALUE,                               \
                                THDS_PER_BLOCK,                               \
                                stream,                                       \
                                load_func,                                    \
                                store_func,                                   \
                                cache_kv_I,                                   \
                                cache_k_quant_scale,                          \
                                cache_v_quant_scale,                          \
                                cache_k_dequant_scale,                        \
                                cache_v_dequant_scale)                        \
  size_t smem_sz =                                                            \
      smem_size_in_bytes<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK);      \
  dim3 grid(params.num_head, params.batch_size);                              \
  constexpr auto kernel_fn =                                                  \
      masked_multihead_attention_kernel_int8<T,                               \
                                             Dh,                              \
                                             Dh_MAX,                          \
                                             THDS_PER_KEY,                    \
                                             THDS_PER_VALUE,                  \
                                             THDS_PER_BLOCK,                  \
                                             decltype(load_func),             \
                                             decltype(store_func)>;           \
  if (smem_sz > 0xc000) {                                                     \
    cudaFuncSetAttribute(                                                     \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);     \
  }                                                                           \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params,                \
                                                       load_func,             \
                                                       store_func,            \
                                                       cache_kv_I,            \
                                                       cache_k_quant_scale,   \
                                                       cache_v_quant_scale,   \
                                                       cache_k_dequant_scale, \
                                                       cache_v_dequant_scale);

#define MMHA_LAUNCH_KERNEL(T,                                             \
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
  dim3 grid(params.num_head, params.batch_size);                          \
  constexpr auto kernel_fn =                                              \
      masked_multihead_attention_kernel<T,                                \
                                        Dh,                               \
                                        Dh_MAX,                           \
                                        THDS_PER_KEY,                     \
                                        THDS_PER_VALUE,                   \
                                        THDS_PER_BLOCK,                   \
                                        decltype(load_func),              \
                                        decltype(store_func)>;            \
  if (smem_sz > 0xc000) {                                                 \
    cudaFuncSetAttribute(                                                 \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz); \
  }                                                                       \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(                   \
      params, load_func, store_func);

template <typename T,
          int Dh,
          int Dh_MAX,
          typename LoadFunc,
          typename StoreFunc,
          int BlockSizeMax,
          int BlockSizeMiddle = BlockSizeMax>
void fmha_launch_kernel_impl_int8(
    const Masked_multihead_attention_params<T> &params,
    const cudaStream_t &stream,
    LoadFunc load_func,
    StoreFunc store_func,
    uint8_t *cache_kv_I,
    float cache_k_quant_scale,
    float cache_v_quant_scale,
    float cache_k_dequant_scale,
    float cache_v_dequant_scale) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  if (params.timestep < 32) {
    MMHA_LAUNCH_KERNEL_INT8(T,
                            Dh,
                            Dh_MAX,
                            4,
                            THREADS_PER_VALUE,
                            256,
                            stream,
                            load_func,
                            store_func,
                            cache_kv_I,
                            cache_k_quant_scale,
                            cache_v_quant_scale,
                            cache_k_dequant_scale,
                            cache_v_dequant_scale);
  } else if (params.timestep < 2048) {
#if defined(MMHA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
    MMHA_LAUNCH_KERNEL_INT8(T,
                            Dh,
                            Dh_MAX,
                            4,
                            THREADS_PER_VALUE,
                            BlockSizeMiddle,
                            stream,
                            load_func,
                            store_func,
                            cache_kv_I,
                            cache_k_quant_scale,
                            cache_v_quant_scale,
                            cache_k_dequant_scale,
                            cache_v_dequant_scale);
#else
    MMHA_LAUNCH_KERNEL_INT8(T,
                            Dh,
                            Dh_MAX,
                            4,
                            THREADS_PER_VALUE,
                            BlockSizeMiddle,
                            stream,
                            load_func,
                            store_func,
                            cache_kv_I,
                            cache_k_quant_scale,
                            cache_v_quant_scale,
                            cache_k_dequant_scale,
                            cache_v_dequant_scale);
#endif
  } else {
    MMHA_LAUNCH_KERNEL_INT8(T,
                            Dh,
                            Dh_MAX,
                            4,
                            THREADS_PER_VALUE,
                            BlockSizeMax,
                            stream,
                            load_func,
                            store_func,
                            cache_kv_I,
                            cache_k_quant_scale,
                            cache_v_quant_scale,
                            cache_k_dequant_scale,
                            cache_v_dequant_scale);
  }
}

template <typename T, int Dh, int Dh_MAX, typename LoadFunc, typename StoreFunc>
void fmha_launch_kernel_impl(const Masked_multihead_attention_params<T> &params,
                             const cudaStream_t &stream,
                             LoadFunc load_func,
                             StoreFunc store_func) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  if (params.timestep < 32) {
    MMHA_LAUNCH_KERNEL(
        T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, stream, load_func, store_func);
  } else if (params.timestep < 2048) {
#if defined(MMHA_USE_HMMA_FOR_REDUCTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 750
    MMHA_LAUNCH_KERNEL(T,
                       Dh,
                       Dh_MAX,
                       4,
                       THREADS_PER_VALUE,
                       256,
                       stream,
                       load_func,
                       store_func);
#else
    MMHA_LAUNCH_KERNEL(T,
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
    MMHA_LAUNCH_KERNEL(T,
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

template <typename T,
          int Dh,
          int Dh_MAX,
          typename LoadFunc,
          typename StoreFunc,
          bool WITH_INT8 = false>
void fmha_launch_kernel(const Masked_multihead_attention_params<T> &params,
                        const cudaStream_t &stream,
                        LoadFunc load_func,
                        StoreFunc store_func,
                        uint8_t *cache_kv_I,
                        float cache_k_quant_scale,
                        float cache_v_quant_scale,
                        float cache_k_dequant_scale,
                        float cache_v_dequant_scale) {
  if (WITH_INT8) {
    int dev = 0;
    int sm_count = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (params.num_head * params.batch_size <= sm_count) {
      fmha_launch_kernel_impl_int8<T, Dh, Dh_MAX, LoadFunc, StoreFunc, 1024>(
          params,
          stream,
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
    } else if (params.batch_size) {
      fmha_launch_kernel_impl_int8<T, Dh, Dh_MAX, LoadFunc, StoreFunc, 256>(
          params,
          stream,
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
    }
  } else {
    fmha_launch_kernel_impl<T, Dh, Dh_MAX, LoadFunc, StoreFunc>(
        params, stream, load_func, store_func);
  }
}

template <typename T, typename LoadFunc, typename StoreFunc, bool WITH_INT8>
void fmha_impl(const phi::GPUContext &dev_ctx,
               const Masked_multihead_attention_params<T> &params,
               int dim_head,
               LoadFunc load_func,
               StoreFunc store_func,
               uint8_t *cache_kv_I,
               float cache_k_quant_scale,
               float cache_v_quant_scale,
               float cache_k_dequant_scale,
               float cache_v_dequant_scale) {
  switch (dim_head) {
    case 10:
      fmha_launch_kernel<T, 10, 32, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    case 26:
      fmha_launch_kernel<T, 26, 32, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    case 32:
      fmha_launch_kernel<T, 32, 32, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    case 64:
      fmha_launch_kernel<T, 64, 64, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    case 96:
      fmha_launch_kernel<T, 96, 128, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    case 128:
      fmha_launch_kernel<T, 128, 128, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    case 192:
      fmha_launch_kernel<T, 192, 256, LoadFunc, StoreFunc, WITH_INT8>(
          params,
          dev_ctx.stream(),
          load_func,
          store_func,
          cache_kv_I,
          cache_k_quant_scale,
          cache_v_quant_scale,
          cache_k_dequant_scale,
          cache_v_dequant_scale);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Dim_head = %d is unsupport!", dim_head));
  }
}

template <typename T, bool CACHE_KV_INT8>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const Masked_multihead_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  uint8_t *cache_kv_I,
                  float cache_k_quant_scale,
                  float cache_v_quant_scale,
                  float cache_k_dequant_scale,
                  float cache_v_dequant_scale) {
  MMHALoad<T> load_func(qkv_tensor.data<T>());
  MMHAStore<T> store_func(out_tensor->data<T>());
  fmha_impl<T, decltype(load_func), decltype(store_func), CACHE_KV_INT8>(
      dev_ctx,
      params,
      dim_head,
      load_func,
      store_func,
      cache_kv_I,
      cache_k_quant_scale,
      cache_v_quant_scale,
      cache_k_dequant_scale,
      cache_v_dequant_scale);
}

template <typename T, bool CACHE_KV_INT8>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const phi::DenseTensor &shift,
                  const phi::DenseTensor &smooth,
                  const Masked_multihead_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const phi::DenseTensor *dequant_qkv_scales = nullptr,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f,
                  uint8_t *cache_kv_I = nullptr,
                  float cache_k_quant_scale = -1.0f,
                  float cache_v_quant_scale = -1.0f,
                  float cache_k_dequant_scale = -1.0f,
                  float cache_v_dequant_scale = -1.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          num_head * dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    fmha_impl<T, decltype(load_func), decltype(store_func), CACHE_KV_INT8>(
        dev_ctx,
        params,
        dim_head,
        load_func,
        store_func,
        cache_kv_I,
        cache_k_quant_scale,
        cache_v_quant_scale,
        cache_k_dequant_scale,
        cache_v_dequant_scale);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          num_head * dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    fmha_impl<T, decltype(load_func), decltype(store_func), CACHE_KV_INT8>(
        dev_ctx,
        params,
        dim_head,
        load_func,
        store_func,
        cache_kv_I,
        cache_k_quant_scale,
        cache_v_quant_scale,
        cache_k_dequant_scale,
        cache_v_dequant_scale);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     num_head * dim_head);
    fmha_impl<T, decltype(load_func), decltype(store_func), CACHE_KV_INT8>(
        dev_ctx,
        params,
        dim_head,
        load_func,
        store_func,
        cache_kv_I,
        cache_k_quant_scale,
        cache_v_quant_scale,
        cache_k_dequant_scale,
        cache_v_dequant_scale);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     num_head * dim_head);
    fmha_impl<T, decltype(load_func), decltype(store_func), CACHE_KV_INT8>(
        dev_ctx,
        params,
        dim_head,
        load_func,
        store_func,
        cache_kv_I,
        cache_k_quant_scale,
        cache_v_quant_scale,
        cache_k_dequant_scale,
        cache_v_dequant_scale);
  }
}

template <typename T>
void fmha(const phi::GPUContext &dev_ctx,
          const phi::DenseTensor &qkv_tensor,
          const phi::DenseTensor &qkv_bias_tensor,
          const phi::DenseTensor *src_mask_tensor,
          const phi::DenseTensor *cum_offsets_tensor,
          const phi::DenseTensor *sequence_lengths_tensor,
          const phi::DenseTensor *rotary_tensor,
          const phi::DenseTensor *beam_cache_offset_tensor,
          phi::DenseTensor *cache_kv_tensor,
          phi::DenseTensor *out_tensor,
          int batch_size,
          int cache_batch_size,
          int seq_len,
          int max_seq_length,
          int num_head,
          int dim_head,
          int timestep,
          int rotary_emb_dims,
          float inv_sqrt_dh,
          const bool mask_broadcast_num_heads = true,
          const bool add_qkv_bias = true,
          const bool neox_rotary_style = false,
          const phi::DenseTensor *dequant_qkv_scales = nullptr,
          const phi::DenseTensor *shift = nullptr,
          const phi::DenseTensor *smooth = nullptr,
          const float cache_k_quant_scale = -1.0,
          const float cache_v_quant_scale = -1.0,
          const float cache_k_dequant_scale = -1.0,
          const float cache_v_dequant_scale = -1.0,
          const float quant_fmha_out_scale = -1,
          const int quant_round_type = 1,
          const float quant_max_bound = 127.0f,
          const float quant_min_bound = -127.0f,
          const int gqa_group_size = -1) {
  Masked_multihead_attention_params<T> params;
  // params.out = out_tensor->data<T>();
  // params.qkv = qkv_tensor.data<T>();

  if (add_qkv_bias) {
    // Because we may not add qkv_bias, so here we cast to T*.
    // Author(zhengzekang).
    params.qkv_bias = const_cast<T *>(qkv_bias_tensor.data<T>());
  }
  params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  params.cache_kv = cache_kv_tensor->data<T>();

  params.neox_rotary_style = neox_rotary_style;
  if (src_mask_tensor) {
    params.attn_mask = src_mask_tensor->data<T>();
    params.mask_length = src_mask_tensor->dims()[3];
  } else {
    params.attn_mask = nullptr;
    params.mask_length = -1;
  }

  if (sequence_lengths_tensor) {
    params.sequence_lengths = sequence_lengths_tensor->data<int>();
  }

  if (cum_offsets_tensor) {
    params.cum_offsets = cum_offsets_tensor->data<int>();
  } else {
    params.cum_offsets = nullptr;
  }
  params.seq_len = seq_len;

  if (rotary_emb_dims > 0) {
    params.rotary_emb = rotary_tensor->data<float>();
    params.rotary_bsz = rotary_tensor->dims()[1];
  } else {
    params.rotary_emb = nullptr;
    params.rotary_bsz = 0;
  }

  if (beam_cache_offset_tensor) {
    if (cache_k_quant_scale > 0) {
      PADDLE_THROW(phi::errors::Unimplemented(
          "MMHA with int8 cache kv does not support beam search yet"));
    }
    params.beam_cache_offset = beam_cache_offset_tensor->data<int>();
    params.beam_width = beam_cache_offset_tensor->dims()[1];
  }

  if (gqa_group_size > 0) {
    params.gqa_group_size = gqa_group_size;
    params.gqa_num_per_partitions = num_head / gqa_group_size;
  } else {
    params.gqa_group_size = num_head;
    params.gqa_num_per_partitions = 1;
  }

  VLOG(1) << "gqa_group_size " << params.gqa_group_size;
  VLOG(1) << "gqa_num_per_partitions " << params.gqa_num_per_partitions;

  params.add_qkv_bias = add_qkv_bias;
  params.batch_size = batch_size;
  params.cache_batch_size = cache_batch_size;
  params.num_head = num_head;
  params.timestep = timestep;
  params.max_seq_length = max_seq_length;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;

  if (shift != nullptr) {
    if (cache_k_quant_scale > 0) {
      DispatchFMHA<T, true>(dev_ctx,
                            qkv_tensor,
                            *shift,
                            *smooth,
                            params,
                            num_head,
                            dim_head,
                            out_tensor,
                            dequant_qkv_scales,
                            quant_fmha_out_scale,
                            quant_round_type,
                            quant_max_bound,
                            quant_min_bound,
                            cache_kv_tensor->data<uint8_t>(),
                            cache_k_quant_scale,
                            cache_v_quant_scale,
                            cache_k_dequant_scale,
                            cache_v_dequant_scale);
    } else {
      DispatchFMHA<T, false>(dev_ctx,
                             qkv_tensor,
                             *shift,
                             *smooth,
                             params,
                             num_head,
                             dim_head,
                             out_tensor,
                             dequant_qkv_scales,
                             quant_fmha_out_scale,
                             quant_round_type,
                             quant_max_bound,
                             quant_min_bound,
                             nullptr,
                             cache_k_quant_scale,
                             cache_v_quant_scale,
                             cache_k_dequant_scale,
                             cache_v_dequant_scale);
    }
  } else {
    if (cache_k_quant_scale > 0) {
      DispatchFMHA<T, true>(dev_ctx,
                            qkv_tensor,
                            params,
                            num_head,
                            dim_head,
                            out_tensor,
                            cache_kv_tensor->data<uint8_t>(),
                            cache_k_quant_scale,
                            cache_v_quant_scale,
                            cache_k_dequant_scale,
                            cache_v_dequant_scale);
    } else {
      DispatchFMHA<T, false>(dev_ctx,
                             qkv_tensor,
                             params,
                             num_head,
                             dim_head,
                             out_tensor,
                             nullptr,
                             cache_k_quant_scale,
                             cache_v_quant_scale,
                             cache_k_dequant_scale,
                             cache_v_dequant_scale);
    }
  }
}

// NOTE: simd with 16Bytes(128bit), float is 4, float16 is 8
constexpr int VEC_16B = 16;

template <typename T>
__global__ void write_cache_k_kernel(T *cache_k,
                                     const T *k,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int prompt_num,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int seq_len_now = seq_len + prompt_num;
  const int len = seq_lens ? seq_lens[bi] + prompt_num : seq_len_now;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto k_src = reinterpret_cast<const uint4 *>(
      k + bi * num_head * seq_len_now * dim_head + hi * seq_len_now * dim_head);
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

  if (k_seq_len_id < len) {
    k_dst[out_idx] = k_src[k_seq_len_id * dim_head_div_x + k_vec_id];
  }
}

template <typename T>
__global__ void write_cache_v_kernel(T *cache_v,
                                     const T *v,
                                     const int *seq_lens,
                                     const int num_head,
                                     const int dim_head,
                                     const int seq_len,
                                     const int prompt_num,
                                     const int max_seq_len) {
  const int bi = blockIdx.y;
  const int seq_len_now = seq_len + prompt_num;
  const int len = seq_lens ? seq_lens[bi] + prompt_num : seq_len_now;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto v_src = reinterpret_cast<const uint4 *>(
      v + bi * num_head * seq_len_now * dim_head + hi * seq_len_now * dim_head);
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<uint4 *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * len) return;

  v_dst[idx] = v_src[idx];
}

template <typename T, int VecSize>
__forceinline__ __device__ void VectorizedQuant(const T *in,
                                                const float scale,
                                                const int round_type,
                                                const float max_bound,
                                                const float min_bound,
                                                uint8_t *quant_out) {
  phi::AlignedVector<T, VecSize> in_vec{};
  phi::AlignedVector<uint8_t, VecSize> quant_out_vec{};
  phi::Load<T, VecSize>(&in[0], &in_vec);

#pragma unroll
  for (int unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
    float quant_value = scale * static_cast<float>(in_vec[unroll_idx]);
    if (round_type == 0) {
      quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
    } else {
      quant_value = static_cast<float>(round(quant_value));
    }
    quant_value = quant_value > max_bound ? max_bound : quant_value;
    quant_value = quant_value < min_bound ? min_bound : quant_value;
    // TODO(Zhengzekang): if use int4 CacheKV, we may pass a int template named
    // Quantbit, and 128.0f -> (1 << Quantbit)
    quant_out_vec[unroll_idx] = static_cast<uint8_t>(quant_value + 128.0f);
  }
  phi::Store<uint8_t, VecSize>(quant_out_vec, &quant_out[0]);
}

template <typename T>
__global__ void write_cache_k_int8_kernel(uint8_t *cache_k,
                                          const T *k,
                                          const int *seq_lens,
                                          const int num_head,
                                          const int dim_head,
                                          const int seq_len,
                                          const int prompt_num,
                                          const int max_seq_len,
                                          const float scale,
                                          const int round_type,
                                          const float max_bound,
                                          const float min_bound) {
  const int bi = blockIdx.y;
  const int seq_len_now = seq_len + prompt_num;
  const int len = seq_lens ? seq_lens[bi] + prompt_num : seq_len_now;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;
  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  using Packed_Int8_t = typename packed_type<uint8_t, X_ELEMS>::type;

  // [bsz, num_head, seq_len, dim_head/x, x]
  auto k_src = reinterpret_cast<const uint4 *>(
      k + bi * num_head * seq_len_now * dim_head + hi * seq_len_now * dim_head);
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  auto k_dst = reinterpret_cast<Packed_Int8_t *>(
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

  if (k_seq_len_id < len) {
    VectorizedQuant<T, X_ELEMS>(
        reinterpret_cast<const T *>(k_src +
                                    (k_seq_len_id * dim_head_div_x + k_vec_id)),
        scale,
        round_type,
        max_bound,
        min_bound,
        reinterpret_cast<uint8_t *>(k_dst + out_idx));
    // k_dst[out_idx] = k_src[k_seq_len_id * dim_head_div_x + k_vec_id];
  }
}

template <typename T>
__global__ void write_cache_v_int8_kernel(uint8_t *cache_v,
                                          const T *v,
                                          const int *seq_lens,
                                          const int num_head,
                                          const int dim_head,
                                          const int seq_len,
                                          const int prompt_num,
                                          const int max_seq_len,
                                          const float scale,
                                          const int round_type,
                                          const float max_bound,
                                          const float min_bound) {
  const int bi = blockIdx.y;
  const int seq_len_now = seq_len + prompt_num;
  const int len = seq_lens ? seq_lens[bi] + prompt_num : seq_len_now;
  if (len == 0) {
    return;
  }

  const int hi = blockIdx.z;

  constexpr int X_ELEMS = VEC_16B / sizeof(T);
  // [bsz, num_head, seq_len, dim_head/x, x]
  using Packed_Int8_t = typename packed_type<uint8_t, X_ELEMS>::type;

  auto v_src = reinterpret_cast<const uint4 *>(
      v + bi * num_head * seq_len_now * dim_head + hi * seq_len_now * dim_head);
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  auto v_dst = reinterpret_cast<Packed_Int8_t *>(
      cache_v + bi * num_head * max_seq_len * dim_head +
      hi * max_seq_len * dim_head);

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const int dim_head_div_x = dim_head / X_ELEMS;

  if (idx >= dim_head_div_x * len) return;

  VectorizedQuant<T, X_ELEMS>(reinterpret_cast<const T *>(v_src + idx),
                              scale,
                              round_type,
                              max_bound,
                              min_bound,
                              reinterpret_cast<uint8_t *>(v_dst + idx));
  // v_dst[idx] = v_src[idx];
}

template <typename T>
void write_int8_cache_kv(const phi::GPUContext &dev_ctx,
                         uint8_t *cache_k,
                         uint8_t *cache_v,
                         const T *k,
                         const T *v,
                         const int *seq_lens,
                         const int bsz,
                         const int num_head,
                         const int seq_len,
                         const int prompt_num,
                         const int max_seq_len,
                         const int dim_head,
                         const int round_type,
                         const float max_bound,
                         const float min_bound,
                         const float cache_k_scale = -1.0,
                         const float cache_v_scale = -1.0) {
  constexpr int block_sz = 128;
  constexpr int x = VEC_16B / sizeof(T);

  assert(dim_head % x == 0);
  PADDLE_ENFORCE_EQ(
      dim_head % x,
      0,
      platform::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, x));

  int max_size = max_seq_len * dim_head / x;
  int size = (seq_len + prompt_num) * dim_head / x;
  dim3 grid(div_up(max_size, block_sz), bsz, num_head);
  dim3 grid_v(div_up(size, block_sz), bsz, num_head);

  // transpose [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  int gk = bsz * num_head * max_seq_len;
  write_cache_k_int8_kernel<<<grid, block_sz, 0, dev_ctx.stream()>>>(
      cache_k,
      k,
      seq_lens,
      num_head,
      dim_head,
      seq_len,
      prompt_num,
      max_seq_len,
      cache_k_scale,
      round_type,
      max_bound,
      min_bound);

  // copy [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  int gv = bsz * num_head * max_seq_len;
  write_cache_v_int8_kernel<<<grid_v, block_sz, 0, dev_ctx.stream()>>>(
      cache_v,
      v,
      seq_lens,
      num_head,
      dim_head,
      seq_len,
      prompt_num,
      max_seq_len,
      cache_v_scale,
      round_type,
      max_bound,
      min_bound);
}

template <typename T>
void write_int8_cache_kv(const phi::GPUContext &dev_ctx,
                         uint8_t *cache_k,
                         uint8_t *cache_v,
                         const T *k,
                         const T *v,
                         const int bsz,
                         const int num_head,
                         const int seq_len,
                         const int max_seq_len,
                         const int dim_head,
                         const int round_type,
                         const float max_bound,
                         const float min_bound,
                         const float cache_k_scale = -1.0,
                         const float cache_v_scale = -1.0) {
  write_int8_cache_kv(dev_ctx,
                      cache_k,
                      cache_v,
                      k,
                      v,
                      nullptr,
                      bsz,
                      num_head,
                      seq_len,
                      0, /*prompt_num*/
                      max_seq_len,
                      dim_head,
                      round_type,
                      max_bound,
                      min_bound,
                      cache_k_scale,
                      cache_v_scale);
}

template <typename T>
void WriteInt8CacheKV(const phi::GPUContext &dev_ctx,
                      const phi::DenseTensor *pre_cache_kv_out,
                      phi::DenseTensor *cache_kv_out,
                      const phi::DenseTensor *kv_transpose_out,
                      const int *sequence_lengths_data,
                      const int cache_bsz,
                      const int bsz,
                      const int num_head,
                      const int seq_len,
                      const int dim_head,
                      const int cache_offset,
                      const int round_type,
                      const float max_bound,
                      const float min_bound,
                      const float cache_k_scale = -1.0,
                      const float cache_v_scale = -1.0) {
  const T *k_ptr = nullptr;
  const T *v_ptr = nullptr;

  if (cache_offset > 0) {
    // [2, bsz, num_head, cache_offset + seq_len, head_dim]
    const T *kv_data = pre_cache_kv_out->data<T>();
    k_ptr = kv_data;
    int64_t k_size = bsz * num_head * (seq_len + cache_offset) * dim_head;
    v_ptr = k_ptr + k_size;
  } else {
    // [3, bsz, num_head, seq_len, head_dim]
    int64_t k_size = bsz * seq_len * num_head * dim_head;
    k_ptr = kv_transpose_out->data<T>();
    v_ptr = k_ptr + k_size;
  }

  // [2, bsz, num_head, max_seq_len, head_dim]
  int max_seq_len = cache_kv_out->dims()[3];
  uint8_t *cache_kv_data = cache_kv_out->data<uint8_t>();
  int64_t cache_k_size = cache_bsz * num_head * max_seq_len * dim_head;

  uint8_t *cache_k_ptr = cache_kv_data;
  uint8_t *cache_v_ptr = cache_kv_data + cache_k_size;

  // const int seq_len_tmp = seq_len + cache_offset;
  write_int8_cache_kv<T>(dev_ctx,
                         cache_k_ptr,
                         cache_v_ptr,
                         k_ptr,
                         v_ptr,
                         sequence_lengths_data,
                         bsz,
                         num_head,
                         seq_len,
                         cache_offset,  // prompt_num
                         max_seq_len,
                         dim_head,
                         round_type,
                         max_bound,
                         min_bound,
                         cache_k_scale,
                         cache_v_scale);
}

template <typename T>
void write_cache_kv(const phi::GPUContext &dev_ctx,
                    T *cache_k,
                    T *cache_v,
                    const T *k,
                    const T *v,
                    const int *seq_lens,
                    const int bsz,
                    const int num_head,
                    const int seq_len,
                    const int prompt_num,
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
  int size = (seq_len + prompt_num) * dim_head / x;
  dim3 grid(div_up(max_size, block_sz), bsz, num_head);
  dim3 grid_v(div_up(size, block_sz), bsz, num_head);

  // transpose [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, dim_head/x, max_seq_len, x]
  write_cache_k_kernel<<<grid, block_sz, 0, dev_ctx.stream()>>>(cache_k,
                                                                k,
                                                                seq_lens,
                                                                num_head,
                                                                dim_head,
                                                                seq_len,
                                                                prompt_num,
                                                                max_seq_len);

  // copy [bsz, num_head, seq_len, dim_head/x, x]->
  // [bsz, num_head, max_seq_len, dim_head/x, x]
  write_cache_v_kernel<<<grid_v, block_sz, 0, dev_ctx.stream()>>>(cache_v,
                                                                  v,
                                                                  seq_lens,
                                                                  num_head,
                                                                  dim_head,
                                                                  seq_len,
                                                                  prompt_num,
                                                                  max_seq_len);
}

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMultiProcessors(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return cudaSuccess;
}

template <typename T, int X_ELEMS>
__global__ void gqa_write_cache_k_kernel(T *cache_k,
                                         const T *k,
                                         const int *seq_lens,
                                         const int *padding_offsets,
                                         const int gqa_group_size,
                                         const int max_seq_len,
                                         const int seq_len,
                                         const int dim_head,
                                         const int64_t num_elems) {
  phi::AlignedVector<T, X_ELEMS> in_vec;

  for (int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * X_ELEMS;
       linear_idx < num_elems;
       linear_idx += blockDim.x * gridDim.x * X_ELEMS) {
    const int hidden_size = gqa_group_size * dim_head;
    const int token_idx = linear_idx / hidden_size;
    const int head_idx = (linear_idx % hidden_size) / dim_head;
    const int head_offset = linear_idx % dim_head;
    const int head_vec_id = head_offset / X_ELEMS;
    const int ori_token_id = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_id / seq_len;

    if (seq_lens[ori_bi] == 0) continue;

    const int local_token_id = ori_token_id % seq_len;

    const int tgt_idx = ori_bi * gqa_group_size * max_seq_len * dim_head +
                        head_idx * max_seq_len * dim_head +
                        head_vec_id * max_seq_len * X_ELEMS +
                        local_token_id * X_ELEMS;

    phi::Load(&k[linear_idx], &in_vec);
    phi::Store(in_vec, &cache_k[tgt_idx]);
  }
}

template <typename T, int X_ELEMS>
__global__ void gqa_write_cache_v_kernel(T *cache_v,
                                         const T *v,
                                         const int *seq_lens,
                                         const int *padding_offsets,
                                         const int gqa_group_size,
                                         const int max_seq_len,
                                         const int seq_len,
                                         const int dim_head,
                                         const int64_t num_elems) {
  phi::AlignedVector<T, X_ELEMS> in_vec;

  for (int64_t linear_idx = (blockIdx.x * blockDim.x + threadIdx.x) * X_ELEMS;
       linear_idx < num_elems;
       linear_idx += blockDim.x * gridDim.x * X_ELEMS) {
    const int hidden_size = gqa_group_size * dim_head;
    const int token_idx = linear_idx / hidden_size;
    const int head_idx = (linear_idx % hidden_size) / dim_head;
    const int head_offset = linear_idx % dim_head;
    const int ori_token_id = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_id / seq_len;

    if (seq_lens[ori_bi] == 0) continue;

    const int local_token_id = ori_token_id % seq_len;

    const int tgt_idx = ori_bi * gqa_group_size * max_seq_len * dim_head +
                        head_idx * max_seq_len * dim_head +
                        local_token_id * dim_head + head_offset;

    phi::Load(&v[linear_idx], &in_vec);
    phi::Store(in_vec, &cache_v[tgt_idx]);
  }
}

template <typename T>
void gqa_write_cachekv(
    const phi::GPUContext &dev_ctx,
    phi::DenseTensor *cache_kv_out,  // [2, cache_bsz, gqa_group_size,
                                     // max_seq_len, dim_head] k need
    const phi::DenseTensor
        &unpadding_k,  // [token_num, gqa_group_size, dim_head]
    const phi::DenseTensor &unpadding_v,
    const phi::DenseTensor &padding_offsets,
    const phi::DenseTensor &seq_lens,
    const int seq_len) {
  constexpr int block_sz = 128;
  constexpr int x = VEC_16B / sizeof(T);

  const int cache_bsz = cache_kv_out->dims()[1];
  const int gqa_group_size = cache_kv_out->dims()[2];
  const int max_seq_len = cache_kv_out->dims()[3];
  const int dim_head = cache_kv_out->dims()[4];

  VLOG(1) << "cache_kv_out->dims() " << cache_kv_out->dims();
  VLOG(1) << "padding_offsets " << padding_offsets;

  assert(dim_head % x == 0);
  PADDLE_ENFORCE_EQ(
      dim_head % x,
      0,
      platform::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, x));

  const int64_t num_elems = unpadding_k.numel();

  T *cache_k = cache_kv_out->data<T>();
  T *cache_v = cache_k + cache_bsz * gqa_group_size * max_seq_len * dim_head;

  int grid_size;
  GetNumBlocks(num_elems, &grid_size);

  gqa_write_cache_k_kernel<T, x><<<grid_size, block_sz, 0, dev_ctx.stream()>>>(
      cache_k,
      unpadding_k.data<T>(),
      seq_lens.data<int>(),
      padding_offsets.data<int>(),
      gqa_group_size,
      max_seq_len,
      seq_len,
      dim_head,
      num_elems);
  gqa_write_cache_v_kernel<T, x><<<grid_size, block_sz, 0, dev_ctx.stream()>>>(
      cache_v,
      unpadding_v.data<T>(),
      seq_lens.data<int>(),
      padding_offsets.data<int>(),
      gqa_group_size,
      max_seq_len,
      seq_len,
      dim_head,
      num_elems);
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
  write_cache_kv(dev_ctx,
                 cache_k,
                 cache_v,
                 k,
                 v,
                 nullptr,
                 bsz,
                 num_head,
                 seq_len,
                 0,
                 max_seq_len,
                 dim_head);
}

template <typename T>
void WriteCacheKV(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor *pre_cache_kv_out,
                  phi::DenseTensor *cache_kv_out,
                  const phi::DenseTensor *kv_transpose_out,
                  const int *sequence_lengths_data,
                  const int cache_bsz,
                  const int bsz,
                  const int num_head,
                  const int seq_len,
                  const int dim_head,
                  const int cache_offset) {
  const T *k_ptr = nullptr;
  const T *v_ptr = nullptr;

  if (cache_offset > 0) {
    // [2, bsz, num_head, cache_offset + seq_len, head_dim]
    const T *kv_data = pre_cache_kv_out->data<T>();
    k_ptr = kv_data;
    int64_t k_size = bsz * num_head * (seq_len + cache_offset) * dim_head;
    v_ptr = k_ptr + k_size;
  } else {
    // [3, bsz, num_head, seq_len, head_dim]
    int64_t k_size = bsz * seq_len * num_head * dim_head;
    k_ptr = kv_transpose_out->data<T>();
    v_ptr = k_ptr + k_size;
  }

  // [2, bsz, num_head, max_seq_len, head_dim]
  int max_seq_len = cache_kv_out->dims()[3];
  T *cache_kv_data = cache_kv_out->data<T>();
  int64_t cache_k_size = cache_bsz * num_head * max_seq_len * dim_head;

  T *cache_k_ptr = cache_kv_data;
  T *cache_v_ptr = cache_kv_data + cache_k_size;

  write_cache_kv<T>(dev_ctx,
                    cache_k_ptr,
                    cache_v_ptr,
                    k_ptr,
                    v_ptr,
                    sequence_lengths_data,
                    bsz,
                    num_head,
                    seq_len,
                    cache_offset,
                    max_seq_len,
                    dim_head);
}

template <typename T, int VecSize>
__global__ void fusedQKV_transpose_split_kernel(T *q_buf,
                                                T *kv_buf,
                                                const T *qkv,
                                                const int *padding_offset,
                                                const int *seq_lens,
                                                const int32_t elem_cnt,
                                                const int batch_size,
                                                const int max_len_this_time,
                                                const int seq_len,
                                                const int token_num,
                                                const int head_num,
                                                const int size_per_head) {
  const int32_t offset =
      batch_size * max_len_this_time * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;
    const int32_t seq_id = ori_token_idx % seq_len;

    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    if (qkv_id == 0) {
      phi::Store<T, VecSize>(
          src_vec,
          &q_buf[target_batch_id * head_num * max_len_this_time *
                     size_per_head +
                 head_id * max_len_this_time * size_per_head +
                 seq_id * size_per_head + size_id]);
    } else {
      const int32_t kv_store_offset = (qkv_id - 1) * offset;
      phi::Store<T, VecSize>(
          src_vec,
          &kv_buf[kv_store_offset +
                  target_batch_id * head_num * max_len_this_time *
                      size_per_head +
                  head_id * max_len_this_time * size_per_head +
                  seq_id * size_per_head + size_id]);
    }
  }
}

template <typename T, int VecSize>
__global__ void fusedQKV_transpose_split_kernel(T *q_buf,
                                                T *k_buf,
                                                T *v_buf,
                                                const T *qkv,
                                                const int *padding_offset,
                                                const int *seq_lens,
                                                const int32_t elem_cnt,
                                                const int batch_size,
                                                const int seq_len,
                                                const int token_num,
                                                const int head_num,
                                                const int size_per_head) {
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;

    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    const int32_t write_idx =
        token_idx * hidden_size + head_id * size_per_head + size_id;
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &q_buf[write_idx]);
    } else if (qkv_id == 1) {
      phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
    }
  }
}

template <typename T, int VecSize, bool ComputeBias>
__global__ void add_fusedQKV_bias_transpose_split_kernel(
    T *q_buf,
    T *kv_buf,
    const T *qkv,
    const T *qkv_bias,
    const int *padding_offset,
    const int32_t elem_cnt,
    const int batch_size,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int size_per_head) {
  const int32_t offset = batch_size * seq_len * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    if (ComputeBias) {
      phi::Load<T, VecSize>(&qkv_bias[bias_idx], &bias_vec);
#pragma unroll
      for (int32_t unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
        src_vec[unroll_idx] += bias_vec[unroll_idx];
      }
    }
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    const int32_t seq_id = ori_token_idx % seq_len;

    // equal to:
    // const int qkv_id  = (linear_index % fused_hidden_size) / hidden_size;
    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    if (qkv_id == 0) {
      phi::Store<T, VecSize>(
          src_vec,
          &q_buf[target_batch_id * head_num * seq_len * size_per_head +
                 head_id * seq_len * size_per_head + seq_id * size_per_head +
                 size_id]);
    } else {
      const int32_t kv_store_offset = (qkv_id - 1) * offset;
      phi::Store<T, VecSize>(
          src_vec,
          &kv_buf[kv_store_offset +
                  target_batch_id * head_num * seq_len * size_per_head +
                  head_id * seq_len * size_per_head + seq_id * size_per_head +
                  size_id]);
    }
  }
}

template <typename T>
void qkv_transpose_split(const phi::GPUContext &dev_ctx,
                         T *q_buf,
                         T *kv_buf,
                         const T *qkv,
                         const int *padding_offset,
                         const int *seq_lens,
                         const int token_num,
                         const int batch_size,
                         const int head_num,
                         const int max_len_this_time,
                         const int seq_len,
                         const int size_per_head) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                      kv_buf,
                                                      qkv,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      max_len_this_time,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head);
}

template <typename T>
void qkv_transpose_split(const phi::GPUContext &dev_ctx,
                         T *q_buf,
                         T *k_buf,
                         T *v_buf,
                         const T *qkv,
                         const int *padding_offset,
                         const int *seq_lens,
                         const int token_num,
                         const int batch_size,
                         const int head_num,
                         const int seq_len,
                         const int size_per_head) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                      k_buf,
                                                      v_buf,
                                                      qkv,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head);
}

template <typename T, int VecSize>
__global__ void gqa_fusedQKV_transpose_split_kernel(T *q_buf,
                                                    T *k_buf,
                                                    T *v_buf,
                                                    const T *qkv,
                                                    const int *padding_offset,
                                                    const int *seq_lens,
                                                    const int32_t elem_cnt,
                                                    const int batch_size,
                                                    const int seq_len,
                                                    const int token_num,
                                                    const int head_num,
                                                    const int size_per_head,
                                                    const int gqa_group_size) {
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  const int fused_hidden_size = (head_num + 2 * gqa_group_size) * size_per_head;

  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    phi::Load<T, VecSize>(&qkv[linear_index], &src_vec);
    int32_t bias_idx = linear_index % fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;

    const int32_t head_id = bias_idx / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    // [token_num, num_head or gqa_group_size, size_per_head]
    if (head_id < head_num) {
      const int32_t write_idx = token_idx * head_num * size_per_head +
                                head_id * size_per_head + size_id;
      phi::Store<T, VecSize>(src_vec, &q_buf[write_idx]);
    } else {
      if (head_id < head_num + gqa_group_size) {
        const int32_t write_idx = token_idx * gqa_group_size * size_per_head +
                                  (head_id - head_num) * size_per_head +
                                  size_id;
        phi::Store<T, VecSize>(src_vec, &k_buf[write_idx]);
      } else {
        const int32_t write_idx =
            token_idx * gqa_group_size * size_per_head +
            (head_id - head_num - gqa_group_size) * size_per_head + size_id;
        phi::Store<T, VecSize>(src_vec, &v_buf[write_idx]);
      }
    }
  }
}

template <typename T>
void gqa_qkv_transpose_split(const phi::GPUContext &dev_ctx,
                             T *q_buf,
                             T *k_buf,
                             T *v_buf,
                             const T *qkv,
                             const int *padding_offset,
                             const int *seq_lens,
                             const int token_num,
                             const int batch_size,
                             const int head_num,
                             const int seq_len,
                             const int size_per_head,
                             const int gqa_group_size) {
  const int32_t elem_cnt =
      token_num * (head_num + 2 * gqa_group_size) * size_per_head;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  gqa_fusedQKV_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                      k_buf,
                                                      v_buf,
                                                      qkv,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head,
                                                      gqa_group_size);
}

template <typename T>
void qkv_bias_add_transpose_split(const phi::GPUContext &dev_ctx,
                                  T *q_buf,
                                  T *kv_buf,
                                  const T *qkv,
                                  const T *qkv_bias,
                                  const int *padding_offset,
                                  const int token_num,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len,
                                  const int size_per_head,
                                  bool compute_bias) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  if (compute_bias) {
    add_fusedQKV_bias_transpose_split_kernel<T, PackSize, true>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                        kv_buf,
                                                        qkv,
                                                        qkv_bias,
                                                        padding_offset,
                                                        elem_cnt,
                                                        batch_size,
                                                        seq_len,
                                                        token_num,
                                                        head_num,
                                                        size_per_head);
  } else {
    add_fusedQKV_bias_transpose_split_kernel<T, PackSize, false>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_buf,
                                                        kv_buf,
                                                        qkv,
                                                        qkv_bias,
                                                        padding_offset,
                                                        elem_cnt,
                                                        batch_size,
                                                        seq_len,
                                                        token_num,
                                                        head_num,
                                                        size_per_head);
  }
}

/* old rope emb */
template <typename T>
__global__ void NeoXRotaryKernel(const T *input,
                                 const float *cos_emb,
                                 const float *sin_emb,
                                 const int *sequence_lengths,
                                 T *output,
                                 const int rotary_emb_dims,
                                 const int batch_size,
                                 const int head_num,
                                 const int seq_len,
                                 const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * head_num * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + ti;
    const int right_idx = base_idx + ti + half_lastdim;
    int emb_idx_left = bi * seq_len * last_dim + si * last_dim + ti;
    int emb_idx_right =
        bi * seq_len * last_dim + si * last_dim + ti + half_lastdim;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);

    float cos_tmp_left = cos_emb[emb_idx_left];
    float sin_tmp_left = sin_emb[emb_idx_left];
    float cos_tmp_right = cos_emb[emb_idx_right];
    float sin_tmp_right = sin_emb[emb_idx_right];

    T res1 =
        static_cast<T>(input_left * cos_tmp_left - input_right * sin_tmp_left);
    T res2 = static_cast<T>(input_right * cos_tmp_right +
                            input_left * sin_tmp_right);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}

template <typename T>
__global__ void RotaryKernel(const T *input,
                             const float *cos_emb,
                             const float *sin_emb,
                             const int *sequence_lengths,
                             T *output,
                             const int rotary_emb_dims,
                             const int batch_size,
                             const int head_num,
                             const int seq_len,
                             const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  // Note(ZhenyuLi): Calculate the relevant data at one time, so that no
  // additional space is required.
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * head_num * seq_len * last_dim +
                   hi * seq_len * last_dim + si * last_dim;
    int left_idx = base_idx + 2 * ti;
    const int right_idx = base_idx + 2 * ti + 1;
    int emb_idx = bi * seq_len * last_dim + si * last_dim + 2 * ti;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);
    float cos_tmp = cos_emb[emb_idx];
    float sin_tmp = sin_emb[emb_idx];
    T res1 = static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
    T res2 = static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}

template <typename T>
__global__ void RotaryKernel(const T *input,
                             const float *cos_emb,
                             const float *sin_emb,
                             const int *sequence_lengths,
                             T *output,
                             const int rotary_emb_dims,
                             const int batch_size,
                             const int head_num,
                             const int max_len_this_time,
                             const int seq_len,
                             const int last_dim) {
  int bi = blockIdx.x;
  int hi = blockIdx.y;
  int si = blockIdx.z;
  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  // Note(ZhenyuLi): Calculate the relevant data at one time, so that no
  // additional space is required.
  for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
    int base_idx = bi * head_num * max_len_this_time * last_dim +
                   hi * max_len_this_time * last_dim + si * last_dim;
    int left_idx = base_idx + 2 * ti;
    const int right_idx = base_idx + 2 * ti + 1;
    int emb_idx = bi * seq_len * last_dim + si * last_dim + 2 * ti;
    float input_left = static_cast<float>(input[left_idx]);
    float input_right = static_cast<float>(input[right_idx]);
    float cos_tmp = cos_emb[emb_idx];
    float sin_tmp = sin_emb[emb_idx];
    T res1 = static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
    T res2 = static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    output[left_idx] = res1;
    output[right_idx] = res2;
  }
}

template <typename T>
void rotary_qk(const phi::GPUContext &dev_ctx,
               T *q,
               T *k,              // kv
               const T *q_input,  // q
               const T *k_input,  // kv
               const float *rotary_emb,
               const int *sequence_lengths,
               const int rotary_emb_dims,
               const int rope_bsz,
               const int batch_size,
               const int head_num,
               const int max_len_this_time,
               const int seq_len,
               const int dim_head) {
  // q_transpose_out_data [bs, head_num, max_len_this_time, dim_head] -> [bs,
  // head_num, max_len_this_time * rotary_emb_dims, dim_head / rotary_emb_dims]
  // kv_transpose_out_data [bs, head_num, max_len_this_time, dim_head] -> [bs,
  // head_num, max_len_this_time * rotary_emb_dims, dim_head / rotary_emb_dims]
  // rotary_emb [2, bs, 1, seq_len, dim_head] -> [2, bs, 1, seq_len *
  // rotary_emb_dims, dim_head / rotary_emb_dims]
  dim3 grid(batch_size, head_num, max_len_this_time * rotary_emb_dims);
  const int last_dim = dim_head / rotary_emb_dims;
  auto getBlockSize = [](int dim) {
    if (dim > 256) {
      return 512;
    } else if (dim > 128) {
      return 256;
    } else if (dim > 64) {
      return 128;
    } else if (dim > 32) {
      return 64;
    } else {
      return 32;
    }
  };
  int BlockSize = getBlockSize(last_dim / 2);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + rope_bsz * seq_len * dim_head;
  RotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
      q_input,
      cos_emb,
      sin_emb,
      sequence_lengths,
      q,
      rotary_emb_dims,
      batch_size,
      head_num,
      max_len_this_time * rotary_emb_dims,
      seq_len * rotary_emb_dims,
      last_dim);
  RotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
      k_input,
      cos_emb,
      sin_emb,
      sequence_lengths,
      k,
      rotary_emb_dims,
      batch_size,
      head_num,
      max_len_this_time * rotary_emb_dims,
      seq_len * rotary_emb_dims,
      last_dim);
}

template <typename T>
void rotary_qk(const phi::GPUContext &dev_ctx,
               T *q,
               T *k,              // kv
               const T *q_input,  // q
               const T *k_input,  // kv
               const float *rotary_emb,
               const int *sequence_lengths,
               const int rotary_emb_dims,
               const int rope_bsz,
               const int batch_size,
               const int head_num,
               const int seq_len,
               const int dim_head,
               const bool neox_rotary_style) {
  // q_transpose_out_data [bs, head_num, seq_len, dim_head] -> [bs, head_num,
  // seq_len * rotary_emb_dims, dim_head / rotary_emb_dims]
  // kv_transpose_out_data [bs, head_num, seq_len, dim_head] -> [bs, head_num,
  // seq_len * rotary_emb_dims, dim_head / rotary_emb_dims] rotary_emb [2, bs,
  // 1, seq_len, dim_head] -> [2, bs, 1, seq_len * rotary_emb_dims, dim_head /
  // rotary_emb_dims]
  dim3 grid(batch_size, head_num, seq_len * rotary_emb_dims);
  const int last_dim = dim_head / rotary_emb_dims;
  auto getBlockSize = [](int dim) {
    if (dim > 256) {
      return 512;
    } else if (dim > 128) {
      return 256;
    } else if (dim > 64) {
      return 128;
    } else if (dim > 32) {
      return 64;
    } else {
      return 32;
    }
  };
  int BlockSize = getBlockSize(last_dim / 2);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + batch_size * seq_len * dim_head;
  if (!neox_rotary_style) {
    RotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        q_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        q,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
    RotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        k_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        k,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
  } else {
    NeoXRotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        q_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        q,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
    NeoXRotaryKernel<<<grid, BlockSize, 0, dev_ctx.stream()>>>(
        k_input,
        cos_emb,
        sin_emb,
        sequence_lengths,
        k,
        rotary_emb_dims,
        batch_size,
        head_num,
        seq_len * rotary_emb_dims,
        last_dim);
  }
}

__global__ void GetPaddingOffset(int *d_token_num,
                                 int *padding_offset,
                                 int *cu_seqlens_data,
                                 const int *sequence_lengths,
                                 const int batch_size,
                                 const int max_seq_len) {
  // get padding offset of each batch
  int total_seq_len = 0;
  int cum_offset = 0;
  int index = 0;
  cu_seqlens_data[0] = 0;
  for (int i = 0; i < batch_size; i++) {
    const int seq_len = sequence_lengths[i];
    for (int j = 0; j < seq_len; j++) {
      padding_offset[index] = cum_offset;
      index++;
    }
    cum_offset += max_seq_len - seq_len;
    total_seq_len += seq_len;
    cu_seqlens_data[i + 1] = cu_seqlens_data[i] + seq_len;
  }
  d_token_num[0] = total_seq_len;
}

void InvokeGetPaddingOffset(const phi::GPUContext &dev_ctx,
                            int *h_token_num,
                            int *d_token_num,
                            int *padding_offset,
                            int *cu_seqlens_data,
                            const int *sequence_lengths,
                            const int batch_size,
                            const int max_seq_len) {
  GetPaddingOffset<<<1, 1, 0, dev_ctx.stream()>>>(d_token_num,
                                                  padding_offset,
                                                  cu_seqlens_data,
                                                  sequence_lengths,
                                                  batch_size,
                                                  max_seq_len);
  memory::Copy(platform::CPUPlace(),
               h_token_num,
               dev_ctx.GetPlace(),
               d_token_num,
               sizeof(int),
               dev_ctx.stream());
}

template <typename T, int VecSize>
__global__ void RebuildPadding(T *output_data,
                               const T *input_data,
                               const int *cum_offsets,
                               const int *seq_len_decoder,
                               const int *seq_len_encoder,
                               const int seq_len,
                               const int dim_embed,
                               const int elem_nums) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / dim_embed;
    const int bias_idx = i % dim_embed;
    int seq_id = 0;
    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] == 0) continue;
    // just encoder or stop, get last token; just decoder, get first token.
    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] != 0)
      seq_id = seq_len_encoder[bi] - 1;
    const int ori_token_idx = bi * seq_len - cum_offsets[bi] + seq_id;
    const int src_offset = ori_token_idx * dim_embed + bias_idx;
    phi::Load<T, VecSize>(&input_data[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &output_data[i]);
  }
}

template <typename T>
void InvokeRebuildPadding(const phi::GPUContext &dev_ctx,
                          T *output_data,
                          const T *input_data,
                          const int *cum_offsets,
                          const int *seq_len_decoder,
                          const int *seq_len_encoder,
                          const int seq_len,
                          const int token_num,
                          const int dim_embed,
                          const int64_t elem_nums) {
  // src: [token_num, dim_embed]
  // dst: [batch_size, 1, dim_embed]
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(dim_embed % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_embed=%d must be divisible by vec_size=%d",
                        dim_embed,
                        PackSize));
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  RebuildPadding<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(output_data,
                                                      input_data,
                                                      cum_offsets,
                                                      seq_len_decoder,
                                                      seq_len_encoder,
                                                      seq_len,
                                                      dim_embed,
                                                      elem_nums);
}

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return max(a, b);
  }
};

template <int THREADBLOCK_SIZE>
__global__ void GetMaxLenKernel(const int *seq_lens,
                                int *max_len,
                                const int batch_size) {
  const int tid = threadIdx.x;

  typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int max_len_this_thread = 0;
  for (int i = tid; i < batch_size; i += blockDim.x) {
    max_len_this_thread = max(seq_lens[i], max_len_this_thread);
  }
  int total =
      BlockReduce(temp_storage).Reduce(max_len_this_thread, MaxOp<int>());
  if (tid == 0) {
    *max_len = total;
  }
}

int GetMaxLen(const phi::GPUContext &dev_ctx,
              const phi::DenseTensor &seq_lens_tensor,
              phi::DenseTensor *max_len_tensor,
              const int batch_size) {
  constexpr int blockSize = 128;
  int max_len_cpu = 0;
  GetMaxLenKernel<blockSize><<<1, blockSize, 0, dev_ctx.stream()>>>(
      seq_lens_tensor.data<int>(), max_len_tensor->data<int>(), batch_size);
  memory::Copy(platform::CPUPlace(),
               &max_len_cpu,
               dev_ctx.GetPlace(),
               max_len_tensor->data<int>(),
               sizeof(int),
               dev_ctx.stream());
  return max_len_cpu;
}

template <typename T, int VecSize>
__global__ void GetDecoderTensorKernel(const T *qkv_out,
                                       const int *cum_offsets,
                                       T *qkv_out_decoder,
                                       const int token_num,
                                       const int batch_size,
                                       const int head_num,
                                       const int seq_len,
                                       const int dim_head,
                                       const int elem_nums) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const int32_t hidden_size = head_num * dim_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / fused_hidden_size;
    const int bias_idx = i % fused_hidden_size;
    const int ori_token_idx = bi * seq_len - cum_offsets[bi];
    const int qkv_id = bias_idx / hidden_size;
    const int head_id = (i % hidden_size) / dim_head;
    const int size_id = i % dim_head;
    const int src_offset = ori_token_idx * fused_hidden_size +
                           qkv_id * hidden_size + head_id * dim_head + size_id;
    phi::Load<T, VecSize>(&qkv_out[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &qkv_out_decoder[i]);
  }
}

template <typename T, int VecSize>
__global__ void GetDecoderRoPEKernel(const T *rope_emb,
                                     T *rope_out_emb,
                                     const int rope_bsz,
                                     const int batch_size,
                                     const int seq_len,
                                     const int dim_head,
                                     const int elem_nums) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  const T *rope_cos_emb = rope_emb;
  const T *rope_sin_emb = rope_emb + rope_bsz * seq_len * dim_head;
  T *cos_emb = rope_out_emb;
  T *sin_emb = rope_out_emb + batch_size * dim_head;
  const int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = global_idx * VecSize; i < elem_nums;
       i += gridDim.x * blockDim.x * VecSize) {
    const int bi = i / dim_head;
    const int src_offset = bi * seq_len * dim_head + i % dim_head;
    phi::Load<T, VecSize>(&rope_cos_emb[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &cos_emb[i]);
    phi::Load<T, VecSize>(&rope_sin_emb[src_offset], &src_vec);
    phi::Store<T, VecSize>(src_vec, &sin_emb[i]);
  }
}

template <typename T>
void GetDecoderTensor(const phi::GPUContext &dev_ctx,
                      const phi::DenseTensor &qkv_out,
                      const phi::DenseTensor *rope_emb,
                      const int *cum_offsets,
                      phi::DenseTensor *qkv_out_decoder,
                      phi::DenseTensor *rope_out_emb,
                      const int token_num,
                      const int batch_size,
                      const int num_head,
                      const int seq_len,
                      const int dim_head) {
  // qkv_out: [token_num, 3, num_head, dim_head] -> [bs, 1, 3, num_head,
  // dim_head] rope: [2, bsz, 1, seq_len, dim_head] -> [2, bsz, 1, 1, dim_head]
  int elem_nums = qkv_out_decoder->numel();
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      dim_head % PackSize,
      0,
      platform::errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", dim_head, PackSize));
  int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  GetDecoderTensorKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          qkv_out.data<T>(),
          cum_offsets,
          qkv_out_decoder->data<T>(),
          token_num,
          batch_size,
          num_head,
          seq_len,
          dim_head,
          elem_nums);
  if (rope_out_emb) {
    elem_nums = rope_out_emb->numel() / 2;
    pack_num = elem_nums / PackSize;
    GetNumBlocks(pack_num, &grid_size);
    GetDecoderRoPEKernel<float, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            rope_emb->data<float>(),
            rope_out_emb->data<float>(),
            rope_emb->dims()[1],
            batch_size,
            seq_len,
            dim_head,
            elem_nums);
  }
}

template <typename T>
__global__ void RemovePadding(T *output_data,
                              const T *input_data,
                              const int *padding_offset,
                              const int dim_embed) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int src_seq_id = bid + padding_offset[bid];
  const int tgt_seq_id = bid;

  for (int i = tid; i < dim_embed; i += blockDim.x) {
    output_data[tgt_seq_id * dim_embed + i] =
        input_data[src_seq_id * dim_embed + i];
  }
}

template <typename T>
void InvokeRemovePadding(const phi::GPUContext &dev_ctx,
                         T *output_data,
                         const T *input_data,
                         const int *padding_offset,
                         const int token_num,
                         const int dim_embed) {
  RemovePadding<<<token_num, 256, 0, dev_ctx.stream()>>>(
      output_data, input_data, padding_offset, dim_embed);
}

template <typename T>
__global__ void RebuildPadding(T *output_data,
                               const T *input_data,
                               const int *padding_offset,
                               const int dim_embed) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int dst_seq_id = bid + padding_offset[bid];
  const int src_seq_id = bid;

  for (int i = tid; i < dim_embed; i += blockDim.x) {
    output_data[dst_seq_id * dim_embed + i] =
        input_data[src_seq_id * dim_embed + i];
  }
}

template <typename T>
void InvokeRebuildPadding(const phi::GPUContext &dev_ctx,
                          T *output_data,
                          const T *input_data,
                          const int *padding_offset,
                          const int token_num,
                          const int dim_embed) {
  // src: [token_num, dim_embed]
  // dst: [batch_size * max_seq_len, dim_embed]
  RebuildPadding<<<token_num, 256, 0, dev_ctx.stream()>>>(
      output_data, input_data, padding_offset, dim_embed);
}

template <typename T, int VecSize>
__global__ void InitOutValueKernel(T *output_data,
                                   const int64_t numel,
                                   const T init_value) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  int64_t global_thread_idx = bid * blockDim.x + tid;

  for (int linear_index = global_thread_idx * VecSize,
           step = gridDim.x * blockDim.x * VecSize;
       linear_index < numel;
       linear_index += step) {
    for (int i = 0; i < VecSize; i++) {
      output_data[linear_index + i] = init_value;
    }
  }
}

template <typename T>
void InitValue(const phi::GPUContext &dev_ctx,
               T *output_data,
               const int64_t numel,
               const T init_value) {
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      numel % PackSize,
      0,
      platform::errors::PreconditionNotMet(
          "numel=%d must be divisible by vec_size=%d", numel, PackSize));
  const int pack_num = numel / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  InitOutValueKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          output_data, numel, init_value);
}

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void ActFFNGlu(const T *bias,
                          Functor act_functor,
                          const int token_num,
                          const int hid_dim,
                          const int elem_num,
                          LoadFunc load_func,
                          StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec1;
  LoadT src_vec2;
  LoadT bias_vec1;
  LoadT bias_vec2;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int bi = i / hid_dim;
    int idx = i % hid_dim;
    // const T *input_this_thread = input + bi * hid_dim * 2;
    // T *output_this_thread = output + bi * hid_dim;
    // phi::Load<T, VecSize>(&input_this_thread[idx], &src_vec1);
    // phi::Load<T, VecSize>(&input_this_thread[idx + hid_dim], &src_vec2);

    load_func.template load<VecSize>(&src_vec1, bi * hid_dim * 2 + idx);
    load_func.template load<VecSize>(&src_vec2,
                                     bi * hid_dim * 2 + idx + hid_dim);

    if (bias) {
      phi::Load<T, VecSize>(&bias[idx], &bias_vec1);
      phi::Load<T, VecSize>(&bias[idx + hid_dim], &bias_vec2);
    }
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if (bias) {
        src_vec1[j] += bias_vec1[j];
        src_vec2[j] += bias_vec2[j];
      }
      src_vec1[j] = act_functor(src_vec1[j]);
      src_vec1[j] *= src_vec2[j];
    }
    // phi::Store<T, VecSize>(src_vec1, &output_this_thread[idx]);
    store_func.template store<VecSize>(src_vec1, bi * hid_dim + idx);
  }
}

template <typename T,
          typename Functor,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void LaunchActFFNGlu(const phi::GPUContext &dev_ctx,
                     const T *bias,
                     const int token_num,
                     const int hid_dim,
                     LoadFunc load_func,
                     StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int elem_cnt = token_num * hid_dim;
  const int blocksize = 128;
  int grid_size = 1;
  Functor functor;
  switch (hid_dim % PackSize) {
    case 0:
      GetNumBlocks(elem_cnt / PackSize, &grid_size);
      ActFFNGlu<T, Functor, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(bias,
                                                          functor,
                                                          token_num,
                                                          hid_dim,
                                                          elem_cnt,
                                                          load_func,
                                                          store_func);
      break;
    default:
      GetNumBlocks(elem_cnt, &grid_size);
      ActFFNGlu<T, Functor, 1><<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          bias, functor, token_num, hid_dim, elem_cnt, load_func, store_func);
      break;
  }
}

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void BiasAct(const T *bias,
                        Functor act_functor,
                        const int rows,
                        const int cols,
                        const int elem_num,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

// Zero Initialize BiasVec.
#pragma unroll
  for (int unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
    bias_vec[unroll_idx] = 0;
  }

  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int row_idx = i / cols;
    int col_idx = i % cols;
    int linear_idx = row_idx * cols + col_idx;
    // phi::Load<T, VecSize>(&input[linear_idx], &src_vec);
    load_func.template load<VecSize>(&src_vec, linear_idx);
    if (bias) {
      phi::Load<T, VecSize>(&bias[col_idx], &bias_vec);
    }
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if (bias) {
        src_vec[j] += bias_vec[j];
      }
      src_vec[j] = act_functor(src_vec[j]);
    }
    // phi::Store<T, VecSize>(src_vec, &output[linear_idx]);
    store_func.template store<VecSize>(src_vec, linear_idx);
  }
}

template <typename T,
          typename Functor,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void LaunchBiasAct(const phi::GPUContext &dev_ctx,
                   const T *bias,
                   const int token_num,
                   const int hid_dim,
                   LoadFunc load_func,
                   StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int elem_cnt = token_num * hid_dim;
  const int blocksize = 128;
  int grid_size = 1;
  Functor functor;
  switch (hid_dim % PackSize) {
    case 0:
      GetNumBlocks(elem_cnt / PackSize, &grid_size);
      BiasAct<T, Functor, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(bias,
                                                          functor,
                                                          token_num,
                                                          hid_dim,
                                                          elem_cnt,
                                                          load_func,
                                                          store_func);
      break;
    default:
      GetNumBlocks(elem_cnt, &grid_size);
      BiasAct<T, Functor, 1><<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          bias, functor, token_num, hid_dim, elem_cnt, load_func, store_func);
      break;
  }
}

template <typename T, int VecSize>
__global__ void fused_transpose_split_kernel(
    T *q_out,           // [total, num_head, head_dim]
    T *k_out,           // [total, num_head, head_dim]
    T *v_out,           // [total, num_head, head_dim]
    const T *q_input,   // [bsz, num_head, seq_len, head_dim]
    const T *kv_input,  // [2, bsz, num_head, seq_len, head_dim]
    const int *padding_offset,
    const int *seq_lens,
    const int32_t elem_cnt,
    const int batch_size,
    const int max_len_this_time,
    const int seq_len,
    const int token_num,
    const int head_num,
    const int size_per_head) {
  const int32_t offset =
      batch_size * max_len_this_time * head_num * size_per_head;
  const int32_t hidden_size = head_num * size_per_head;
  const int32_t fused_hidden_size = 3 * hidden_size;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

  int q_size = token_num * hidden_size;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    int32_t bias_idx = linear_index % fused_hidden_size;
    int32_t current_token = linear_index / fused_hidden_size;
    const int32_t token_idx = linear_index / fused_hidden_size;
    const int32_t ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int32_t target_batch_id = ori_token_idx / seq_len;
    if (seq_lens[target_batch_id] == 0) continue;
    const int32_t seq_id = ori_token_idx % seq_len;

    // equal to:
    // const int qkv_id  = (linear_index % fused_hidden_size) / hidden_size;
    const int32_t qkv_id = bias_idx / hidden_size;
    const int32_t head_id = (linear_index % hidden_size) / size_per_head;
    const int32_t size_id = linear_index % size_per_head;

    if (qkv_id == 0) {  // read q
      phi::Load<T, VecSize>(
          &q_input[target_batch_id * head_num * max_len_this_time *
                       size_per_head +
                   head_id * max_len_this_time * size_per_head +
                   seq_id * size_per_head + size_id],
          &src_vec);
    } else {  // read k/v
      const int32_t kv_store_offset = (qkv_id - 1) * offset;
      phi::Load<T, VecSize>(
          &kv_input[kv_store_offset +
                    target_batch_id * head_num * max_len_this_time *
                        size_per_head +
                    head_id * max_len_this_time * size_per_head +
                    seq_id * size_per_head + size_id],
          &src_vec);
    }
    int32_t write_index =
        linear_index - (qkv_id + 2 * current_token) * hidden_size;
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &q_out[write_index]);
    } else if (qkv_id == 1) {
      phi::Store<T, VecSize>(src_vec, &k_out[write_index]);
    } else if (qkv_id == 2) {
      phi::Store<T, VecSize>(src_vec, &v_out[write_index]);
    }
  }
}

template <typename T>
void TransposeSplit(const phi::GPUContext &dev_ctx,
                    T *q_out,
                    T *k_out,
                    T *v_out,
                    const T *q_input,
                    const T *kv_input,
                    const int *padding_offset,
                    const int *seq_lens,
                    const int token_num,
                    const int batch_size,
                    const int head_num,
                    const int max_len_this_time,
                    const int seq_len,
                    const int size_per_head) {
  const int32_t elem_cnt = token_num * head_num * size_per_head * 3;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(size_per_head % PackSize,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dim_head=%d must be divisible by vec_size=%d",
                        size_per_head,
                        PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t blocksize = 128;
  int32_t grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  fused_transpose_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(q_out,
                                                      k_out,
                                                      v_out,
                                                      q_input,
                                                      kv_input,
                                                      padding_offset,
                                                      seq_lens,
                                                      elem_cnt,
                                                      batch_size,
                                                      max_len_this_time,
                                                      seq_len,
                                                      token_num,
                                                      head_num,
                                                      size_per_head);
}

template <typename T>
void TransposeSplit(const phi::GPUContext &dev_ctx,
                    T *q_out,
                    T *k_out,
                    T *v_out,
                    const T *q_input,
                    const T *kv_input,
                    const int *padding_offset,
                    const int *seq_lens,
                    const int token_num,
                    const int batch_size,
                    const int head_num,
                    const int seq_len,
                    const int size_per_head) {
  TransposeSplit<T>(dev_ctx,
                    q_out,
                    k_out,
                    v_out,
                    q_input,
                    kv_input,
                    padding_offset,
                    seq_lens,
                    token_num,
                    batch_size,
                    head_num,
                    seq_len,
                    seq_len,
                    size_per_head);
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 2 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t base_idx = token_idx * 3 * hidden_size +
                             qkv_id * hidden_size + hi * last_dim + h_bias;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                   // [token_num, 3, num_head, dim_head]
    const T *qkv_input,       // qkv
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head) {
  const int elem_nums = token_num * 2 * head_num * dim_head;  // just q and k
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  VariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head);
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const int gqa_group_size) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (num_head + gqa_group_size) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    // [token_num, num_head + 2 * gqa_group_size, last_dim]
    const int64_t base_idx =
        token_idx * (num_head + 2 * gqa_group_size) * last_dim + hi * last_dim +
        h_bias;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left = static_cast<float>(src_vec[2 * i]);
      const float input_right = static_cast<float>(src_vec[2 * i + 1]);
      const float cos_tmp = cos_emb_vec[i];
      const float sin_tmp = sin_emb_vec[i];
      src_vec[2 * i] =
          static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
      src_vec[2 * i + 1] =
          static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                   // [token_num, 3, num_head, dim_head]
    const T *qkv_input,       // qkv
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int gqa_group_size) {
  const int elem_nums =
      token_num * (head_num + gqa_group_size) * dim_head;  // just q and k
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  GQAVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head,
                                                      gqa_group_size);
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<int, VecSize>;
  using LoadBiasT = phi::AlignedVector<T, VecSize>;
  using LoadScaleT = phi::AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    phi::Load<int, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (qkv_id < 2) {
      phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = input_left * out_scale_vec[2 * i] +
                   static_cast<float>(bias_vec[2 * i]);
      input_right = input_right * out_scale_vec[2 * i + 1] +
                    static_cast<float>(bias_vec[2 * i + 1]);
      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                       // [token_num, 3, num_head, dim_head]
    const int *qkv_input,         // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head) {
  const int elem_nums = token_num * 3 * head_num * dim_head;  // just q and k
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  VariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv_out_scales,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head);
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const int *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_biases,          // [3, num_head, dim_head]
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const int gqa_group_size) {
  using LoadT = phi::AlignedVector<int, VecSize>;
  using LoadBiasT = phi::AlignedVector<T, VecSize>;
  using LoadScaleT = phi::AlignedVector<float, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, HalfVecSize>;
  LoadT src_vec;
  LoadBiasT bias_vec;
  LoadScaleT out_scale_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int offset = (num_head + 2 * gqa_group_size) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx = ori_seq_id * half_lastdim + h_bias / 2;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    phi::Load<int, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&qkv_out_scales[bias_idx], &out_scale_vec);
    if (hi < num_head + gqa_group_size) {
      phi::Load<float, HalfVecSize>(&cos_emb[emb_idx], &cos_emb_vec);
      phi::Load<float, HalfVecSize>(&sin_emb[emb_idx], &sin_emb_vec);
    }
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      float input_left = static_cast<float>(src_vec[2 * i]);
      float input_right = static_cast<float>(src_vec[2 * i + 1]);
      // dequant + bias_add
      input_left = input_left * out_scale_vec[2 * i] +
                   static_cast<float>(bias_vec[2 * i]);
      input_right = input_right * out_scale_vec[2 * i + 1] +
                    static_cast<float>(bias_vec[2 * i + 1]);
      if (hi < num_head + gqa_group_size) {  // qk rope
        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        bias_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        bias_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        bias_vec[2 * i] = static_cast<T>(input_left);
        bias_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(bias_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,                       // [token_num, 3, num_head, dim_head]
    const int *qkv_input,         // qkv
    const float *qkv_out_scales,  // [3, num_head, dim_head]
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int gqa_group_size) {
  const int elem_nums =
      token_num * (head_num + 2 * gqa_group_size) * dim_head;  // for all q k v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
  GQAVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv_out_scales,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head,
                                                      gqa_group_size);
}

template <typename T, int VecSize = 1>
__global__ void VariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  const int hidden_size = num_head * last_dim;
  const int offset = 3 * hidden_size;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / last_dim;
    const int h_bias = qkv_bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx =
        ori_bi * seq_len * last_dim + ori_seq_id * last_dim + h_bias;
    const int64_t bias_idx = qkv_id * hidden_size + hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * 3 * hidden_size + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);

      if (qkv_id < 2) {  // qk rope
        const float cos_tmp = cos_emb_vec[2 * i];
        const float sin_tmp = sin_emb_vec[2 * i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, bs, 1, seq_len, dim_head]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int rope_bsz) {
  const int elem_nums = token_num * 3 * head_num * dim_head;  // just q and k
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + rope_bsz * input_output_len * dim_head;

  VariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head);
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotaryKernel(
    const T *qkv,
    const float *cos_emb,  // [1, 1, seq_len, dim_head / 2]
    const float *sin_emb,
    const int *padding_offsets,
    const int *seq_lens,
    const T *qkv_biases,
    T *qkv_out,
    const int64_t elem_cnt,
    const int num_head,
    const int seq_len,
    const int last_dim,
    const int gqa_group_size) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = phi::AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int half_lastdim = last_dim / 2;
  // const int hidden_size = num_head * last_dim;
  const int offset = (num_head + 2 * gqa_group_size) * last_dim;
  for (int64_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int bias = linear_index % offset;
    const int hi = bias / last_dim;
    const int h_bias = bias % last_dim;

    const int ori_seq_id = ori_token_idx % seq_len;

    const int64_t emb_idx =
        ori_bi * seq_len * last_dim + ori_seq_id * last_dim + h_bias;
    const int64_t bias_idx = hi * last_dim + h_bias;
    const int64_t base_idx = token_idx * offset + bias_idx;
    phi::Load<T, VecSize>(&qkv[base_idx], &src_vec);
    phi::Load<T, VecSize>(&qkv_biases[bias_idx], &bias_vec);
    phi::Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
    phi::Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
    for (int i = 0; i < HalfVecSize; i++) {
      const float input_left =
          static_cast<float>(src_vec[2 * i] + bias_vec[2 * i]);
      const float input_right =
          static_cast<float>(src_vec[2 * i + 1] + bias_vec[2 * i + 1]);
      // const float cos_tmp = cos_emb_vec[i];
      // const float sin_tmp = sin_emb_vec[i];
      // src_vec[2 * i] = static_cast<T>(input_left * cos_tmp - input_right *
      // sin_tmp); src_vec[2 * i + 1] = static_cast<T>(input_right * cos_tmp +
      // input_left * sin_tmp);

      if (hi < num_head + gqa_group_size) {  // qk rope
        const float cos_tmp = cos_emb_vec[2 * i];
        const float sin_tmp = sin_emb_vec[2 * i];
        src_vec[2 * i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec[2 * i + 1] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      } else {
        src_vec[2 * i] = static_cast<T>(input_left);
        src_vec[2 * i + 1] = static_cast<T>(input_right);
      }
    }
    phi::Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
  }
}

template <typename T>
void gqa_rotary_qk_variable(
    const phi::GPUContext &dev_ctx,
    T *qkv,              // [token_num, 3, num_head, dim_head]
    const T *qkv_input,  // qkv
    const T *qkv_bias,
    const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
    const int *padding_offsets,
    const int *seq_lens,
    const int token_num,
    const int head_num,
    const int seq_len,
    const int input_output_len,
    const int dim_head,
    const int gqa_group_size,
    const int rope_bsz) {
  const int elem_nums =
      token_num * (head_num + 2 * gqa_group_size) * dim_head;  // for all q k v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + rope_bsz * input_output_len * dim_head;
  GQAVariableLengthRotaryKernel<T, PackSize>
      <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(qkv_input,
                                                      cos_emb,
                                                      sin_emb,
                                                      padding_offsets,
                                                      seq_lens,
                                                      qkv_bias,
                                                      qkv,
                                                      elem_nums,
                                                      head_num,
                                                      seq_len,
                                                      dim_head,
                                                      gqa_group_size);
}

template <typename T, int VecSize = 1>
__global__ void cache_kernel(
    const T *__restrict__ qkv,  // [num_tokens, num_heads + 2 * gqa_group_size,
                                // head_size]
    T *__restrict__ key_cache,  // [num_blocks, gqa_group_size, block_size,
                                // head_size]
    T *__restrict__ value_cache,  // [num_blocks, gqa_group_size, block_size,
                                  // head_size]
    const int *__restrict__ block_tables,     // [bsz, max_blocks_per_seq]
    const int *__restrict__ padding_offsets,  // [num_tokens]
    const int *__restrict__ seq_lens,         // [bsz]
    const int max_seq_len,
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int elem_cnt,
    const int gqa_group_size) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int64_t hidden_size = gqa_group_size * head_size;
  const int64_t offset = 2 * hidden_size;
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / offset;
    const int bias = linear_index % offset;
    const int qkv_id = bias / hidden_size;  // skip q
    const int qkv_bias = bias % hidden_size;
    const int hi = qkv_bias / head_size;
    const int h_bias = qkv_bias % head_size;
    const int ori_token_idx = token_idx + padding_offsets[token_idx];
    const int ori_bi = ori_token_idx / max_seq_len;
    if (seq_lens[ori_bi] == 0) continue;
    const int ori_seq_id = ori_token_idx % max_seq_len;

    const int *block_table_now = block_tables + ori_bi * max_blocks_per_seq;
    const int block_idx = block_table_now[ori_seq_id / block_size];
    const int block_offset = ori_seq_id % block_size;

    const int tgt_idx = block_idx * gqa_group_size * block_size * head_size +
                        hi * block_size * head_size + block_offset * head_size +
                        h_bias;
    const int ori_idx =
        token_idx * (num_heads + 2 * gqa_group_size) * head_size +
        num_heads * head_size + qkv_id * hidden_size + hi * head_size + h_bias;
    phi::Load<T, VecSize>(&qkv[ori_idx], &src_vec);
    if (qkv_id == 0) {
      phi::Store<T, VecSize>(src_vec, &key_cache[tgt_idx]);
    } else {
      phi::Store<T, VecSize>(src_vec, &value_cache[tgt_idx]);
    }
  }
}

template <typename T>
void CacheKernel(const phi::GPUContext &dev_ctx,
                 const phi::DenseTensor
                     &qkv,  // [token_num, 3, num_head, head_dim] ([token_num,
                            // num_head + 2 * gqa_group_size, head_dim] if GQA)
                 const phi::DenseTensor &block_tables,
                 const phi::DenseTensor &padding_offsets,
                 const phi::DenseTensor &seq_lens,
                 const int max_seq_len,
                 phi::DenseTensor *key_cache_out,
                 phi::DenseTensor *value_cache_out,
                 const std::string &cache_quant_type_str,
                 const int num_heads,
                 const int head_size,
                 const int round_type = 0,
                 const bool use_nf4 = false,
                 const float max_bound = 0.0,
                 const float min_bound = 0.0,
                 const int cache_k_group_num = 1,
                 const phi::DenseTensor *cache_k_scales = nullptr,
                 const phi::DenseTensor *cache_v_scales = nullptr,
                 const phi::DenseTensor *cache_k_zero_points = nullptr,
                 const phi::DenseTensor *cache_v_zero_points = nullptr,
                 int gqa_group_size = -1) {
  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  auto qkv_dims = qkv.dims();
  const int max_blocks_per_seq = block_tables.dims()[1];
  const int num_tokens = qkv_dims[0];
  if (gqa_group_size <= 0) {
    gqa_group_size = num_heads;
  }
  const int32_t block_size = key_cache_out->dims()[2];
  const int elem_nums =
      num_tokens * 2 * gqa_group_size * head_size;  // just k and v
  constexpr int PackSize = 16 / sizeof(T);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);

  if (cache_k_scales) {
    PADDLE_THROW(
        phi::errors::Unimplemented("cache kv quant is not supported for now"));
  } else {
    VLOG(1) << "cache kv not quant";
    cache_kernel<DataType_, PackSize>
        <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
            reinterpret_cast<DataType_ *>(const_cast<T *>(qkv.data<T>())),
            reinterpret_cast<DataType_ *>(key_cache_out->data<T>()),
            reinterpret_cast<DataType_ *>(value_cache_out->data<T>()),
            block_tables.data<int>(),
            padding_offsets.data<int>(),
            seq_lens.data<int>(),
            max_seq_len,
            max_blocks_per_seq,
            num_heads,
            head_size,
            block_size,
            elem_nums,
            gqa_group_size);
  }
}

}  // namespace

}  // namespace operators
}  // namespace paddle
