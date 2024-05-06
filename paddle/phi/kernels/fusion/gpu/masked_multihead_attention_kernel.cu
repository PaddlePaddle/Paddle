// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/mmha_util.cu.h"
/// 当前工作是在flashAttention MMHA的基础上，加入一个前处理，使mmha_kernel简洁
// preprocess线程数暂时是错的啦
// Dh_MAX/2在Qk_vec_<float,32>上不能跑，其他也不完全合适
// 还待做的工作：程序边界检查统一/更新注释
namespace phi {
namespace fusion {

#ifndef PADDLE_WITH_HIP

constexpr int steps_per_block = 128;

template <typename T>
struct preprocess_params {
  int kv_num_head;
  int num_head;
  int max_seq_length;
  const int *sequence_lengths{nullptr};
  int timestep;
  bool add_qkv_bias;
  T *qkv_bias;
  bool neox_rotary_style;
  int rotary_emb_dims;
  const float *rotary_emb;
  int batch_size;
  T *cache_kv;
  int cache_batch_size;
  // [B, 1(seq_len), num_head+2*kv_num_head, dim_head]
  T *qkv;
};

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
  T *cache_kv;
  // [B, max_seq_len]
  const int *beam_cache_offset = nullptr;

  const int *sequence_lengths{nullptr};

  // The RoPE embedding, [2, B, rotary_seq_len, 1, dim_head]
  // rotary_emb_dims = 1 if pos_ids_extra is null else 2
  const float *rotary_emb;
  int rotary_emb_dims;
  int rotary_seq_len = 1;

  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;
  int num_head;
  // k_num_head and v_num_head must be equal, we unify them.
  // kv_num_head = k_num_head && kv_num_head == v_num_head
  int kv_num_head;
  int timestep;  // cache_seq_length
  int seq_len;
  int max_seq_length;

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;

  bool add_qkv_bias;
  bool neox_rotary_style;

  float *qk_sum_max_split_seq;
  int *real_split_each_batch;
  int split_seq = -1;
  float *split_out;
};

/// 预取当前KV存入cache中，添加bias并进行位置编码
template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc>
__global__ void preprocess(preprocess_params<T> params, LoadFunc load_func) {
  /// 边界检查
  static_assert(Dh_MAX % 8 == 0, "Dh_MAX % 8 == 0");
  /// 参数准备
  const int bi = blockIdx.y;
  const int hi = blockIdx.x;
  const int tid = threadIdx.x;

  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);
  const int kv_num_head = params.kv_num_head;
  const int num_head = params.num_head;
  int qkv_base_offset = bi * (num_head + 2 * kv_num_head) * Dh + hi * Dh;
  const int num_head_per_group = num_head / kv_num_head;
  const int kv_bhi = bi * kv_num_head + hi / num_head_per_group;
  const int max_seq_length = params.max_seq_length;
  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep
                          : params.sequence_lengths[bi];

  /// 加载当前QK
  using QK_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  constexpr int QK_VECS_IN_16B = 16 / sizeof(QK_vec);
  using QK_vec_RoPE = typename Qk_vec_RoPE_<T, float, Dh_MAX>::Type;
  constexpr int QK_VEC_SIZE = sizeof(QK_vec) / sizeof(T);
  constexpr int QK_VECS_PER_BLK = Dh_MAX / QK_VEC_SIZE;

  T *q_bias_base = nullptr;
  T *k_bias_base = nullptr;
  if (params.add_qkv_bias) {
    q_bias_base = params.qkv_bias;
    k_bias_base = params.qkv_bias + num_head * Dh;
  }
  if (tid < QK_VECS_PER_BLK) {
    QK_vec q;
    zero(q);
    int q_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<QK_vec>(q, q_offset);
    }
    QK_vec k;
    zero(k);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      int k_offset = qkv_base_offset - hi * Dh + num_head * Dh +
                     hi / num_head_per_group * Dh + tid * QK_VEC_SIZE;
      load_func.template load<QK_vec>(k, k_offset);
    }

    int q_bias_offset = hi * Dh + tid * QK_VEC_SIZE;
    int k_bias_offset = hi / num_head_per_group * Dh + tid * QK_VEC_SIZE;
    if (params.add_qkv_bias) {
      QK_vec q_bias;
      zero(q_bias);
      QK_vec k_bias;
      zero(k_bias);
      q_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const QK_vec *>(&q_bias_base[q_bias_offset])
              : q_bias;
      k_bias =
          (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
              ? *reinterpret_cast<const QK_vec *>(&k_bias_base[k_bias_offset])
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
        QK_vec_RoPE cos_emb, sin_emb;
        zero(cos_emb);
        zero(sin_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const QK_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const QK_vec_RoPE *>(
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
        int q_right_bias_offset = hi * Dh + right_id * QK_VEC_SIZE;
        int k_right_bias_offset =
            hi / num_head_per_group * Dh + right_id * QK_VEC_SIZE;
        QK_vec q_right;
        zero(q_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          load_func.template load<QK_vec>(q_right, qk_right_offset);
        }
        QK_vec k_right;
        zero(k_right);
        if (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh) {
          int k_right_offset = num_head * Dh + qk_right_offset - hi * Dh +
                               hi / num_head_per_group * Dh;
          load_func.template load<QK_vec>(k_right, k_right_offset);
        }
        if (params.add_qkv_bias) {
          QK_vec q_right_bias;
          zero(q_right_bias);
          q_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
                             ? *reinterpret_cast<const QK_vec *>(
                                   &q_bias_base[q_right_bias_offset])
                             : q_right_bias;
          QK_vec k_right_bias;
          zero(k_right_bias);
          k_right_bias = (Dh == Dh_MAX || right_id * QK_VEC_SIZE < Dh)
                             ? *reinterpret_cast<const QK_vec *>(
                                   &k_bias_base[k_right_bias_offset])
                             : k_right_bias;
          q_right = add(q_right, q_right_bias);
          k_right = add(k_right, k_right_bias);
        }

        QK_vec_RoPE cos_emb;
        zero(cos_emb);
        cos_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const QK_vec_RoPE *>(
                            &cos_base[rotary_offset])
                      : cos_emb;
        QK_vec_RoPE sin_emb;
        zero(sin_emb);
        sin_emb = (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh)
                      ? *reinterpret_cast<const QK_vec_RoPE *>(
                            &sin_base[rotary_offset])
                      : sin_emb;
        float alpha = (tid % stride_all_lastdim) < stride
                          ? static_cast<float>(-1)
                          : static_cast<float>(1);
        q = apply_rotary_emb<QK_vec, QK_vec_RoPE>(
            q, q_right, cos_emb, sin_emb, alpha);
        k = apply_rotary_emb<QK_vec, QK_vec_RoPE>(
            k, k_right, cos_emb, sin_emb, alpha);
      }
    }
    /// 将Q存回qkv
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      *reinterpret_cast<QK_vec *>(&params.qkv[q_offset]) = q;
    }
    /// 将K存回cache_k（325行-332行）
    int co = tid / QK_VECS_IN_16B;
    int ci = (tid % QK_VECS_IN_16B) * QK_VEC_SIZE;
    int offset = kv_bhi * max_seq_length * Dh +
                 co * max_seq_length * QK_ELTS_IN_16B +
                 act_time_step * QK_ELTS_IN_16B + ci;
    if (Dh == Dh_MAX || co < Dh / QK_ELTS_IN_16B) {
      *reinterpret_cast<QK_vec *>(&params.cache_kv[offset]) = k;
    }
  }

  /// 加载当前V add_bias 并写回（534行-552行）
  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;
  int v_cache_offset =
      (params.cache_batch_size * kv_num_head + kv_bhi) * max_seq_length * Dh +
      vi;
  T *v_cache = &params.cache_kv[v_cache_offset];
  V_vec v_bias;
  zero(v_bias);
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if ((tid / THREADS_PER_VALUE == 0) && (Dh == Dh_MAX || vi < Dh)) {
    V_vec v;
    int v_offset = qkv_base_offset - hi * Dh + vi +
                   (num_head + kv_num_head + hi / num_head_per_group) * Dh;
    load_func.template load<V_vec>(v, v_offset);
    if (params.add_qkv_bias) {
      int v_bias_offset =
          (kv_num_head + num_head) * Dh + hi / num_head_per_group * Dh + vi;
      v_bias =
          *reinterpret_cast<const V_vec *>(&params.qkv_bias[v_bias_offset]);
      v = add(v, v_bias);
    }
    *reinterpret_cast<V_vec *>(&v_cache[act_time_step * Dh]) = v;
  }
}

template <typename T,
          int Dh,
          int Dh_MAX,
          int THREADS_PER_KEY,
          int THREADS_PER_VALUE,
          int THREADS_PER_BLOCK,
          typename LoadFunc>
__global__ void masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params, LoadFunc load_func) {
/// 程序边界检查
#if CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  const int bi = blockIdx.z;
  if (params.sequence_lengths && params.sequence_lengths[bi] < 0) {
    return;
  }
  static_assert(Dh_MAX % THREADS_PER_KEY == 0, "Dh_MAX % THREADS_PER_KEY == 0");
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0,
                "Dh_MAX % THREADS_PER_VALUE == 0");
  // 读取当前Q_vec或cacheK_vec时，16B最多8个元素。如果要扩展到fp8甚至更小的时候，应该注意！
  static_assert(Dh_MAX % 8 == 0, "Dh_MAX % 8 == 0");
  /// 参数准备
  const int hi = blockIdx.y;
  const int split_index = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int WARP_SIZE = 32;
  constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;

  const int max_seq_length = params.max_seq_length;
  const int kv_num_head = params.kv_num_head;
  const int num_head = params.num_head;
  // qkv [B, S=1, num_head + 2 * kv_num_head, head_dim]
  int qkv_base_offset = bi * (num_head + 2 * kv_num_head) * Dh + hi * Dh;
  int act_time_step = params.sequence_lengths == nullptr
                          ? params.timestep + 1
                          : params.sequence_lengths[bi] +
                                1;  // 表示经过预取后,cache_kv里的k/v数
  const int num_head_per_group = num_head / kv_num_head;
  // cache_k [B, kv_num_head, head_dim/x, max_seq_len, x], x表示16B能存的T的数量
  // cache_v [B, kv_num_head, max_seq_len, head_dim]
  const int kv_bhi = bi * kv_num_head + hi / num_head_per_group;
  constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);  // x

  const int bhi = bi * num_head + hi;
  const int ti =
      params.cum_offsets ? bi * params.seq_len - params.cum_offsets[bi] : -1;
  const int thi = params.cum_offsets ? ti * num_head + hi : -1;
  // for beam
  const int beami = bi % params.beam_width;  // beam id
  const int bbi = bi / params.beam_width;    // real batch id
  const int bbhi = bbi * params.beam_width * num_head + hi;
  const int bi_seq_len_offset = bi * max_seq_length;

  // for split
  int bhsi = bhi * params.split_seq;
  const int start_seq = split_index * steps_per_block;
  if (start_seq >= act_time_step) {
    return;  // 多个batch启动的grid都是一样的，短batch的末尾ThreadBlock会退出。
  }
  int end_seq = start_seq + steps_per_block;
  if (end_seq >= act_time_step) {
    params.real_split_each_batch[bi] = split_index + 1;
    end_seq = act_time_step;
  }

  float qk_max = -FLT_MAX;
  typedef PDDataTypeTraits<T> traits_;
  typedef typename traits_::DataType DataType_;

  // 共享内存
  extern __shared__ char smem_[];
  __shared__ float red_smem[WARPS_PER_BLOCK * 2];
  float *qk_smem = reinterpret_cast<float *>(smem_);
  float *logits_smem = reinterpret_cast<float *>(smem_);
  T *out_smem = reinterpret_cast<T *>(smem_);

  /// 读取当前Q && 当前Q存入sharedMem中(121行-362行)
  // 确保64个线程能读完Q且至少一个线程32bit的情况下，经验性的设置向量尺寸
  using Q_vec = typename Qk_vec_<T, Dh_MAX>::Type;
  __shared__ __align__(sizeof(Q_vec)) T q_smem[Dh_MAX];
  constexpr int QK_VEC_SIZE = sizeof(Q_vec) / sizeof(T);
  constexpr int QK_VECS_PER_BLK = Dh_MAX / QK_VEC_SIZE;
  if (tid < QK_VECS_PER_BLK) {
    int q_offset = qkv_base_offset + tid * QK_VEC_SIZE;
    Q_vec q;
    zero(q);
    if (Dh == Dh_MAX || tid * QK_VEC_SIZE < Dh) {
      load_func.template load<Q_vec>(q, q_offset);
    }
    *reinterpret_cast<Q_vec *>(&q_smem[tid * QK_VEC_SIZE]) = q;
  }
  __syncthreads();

  /// 读取cacheK 顺带读sharedMem中的Q (364行-417行)
  // 保证THREADS_PER_KEY个线程恰好读取一行的16B数据
  using K_vec = typename K_vec_<T, THREADS_PER_KEY>::Type;
  constexpr int K_VEC_SIZE = sizeof(K_vec) / sizeof(T);
  constexpr int K_ELTS_PER_THREAD = Dh_MAX / THREADS_PER_KEY;
  constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;
  static_assert(Dh_MAX == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD, "");

  int ko = tid / THREADS_PER_KEY + start_seq;  // 线程在Key粒度上的分布
  int ki = (tid % THREADS_PER_KEY) * K_VEC_SIZE;  // 线程在元素粒度上的分布
  // cacheK要乘的Q
  K_vec q[K_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < K_VECS_PER_THREAD; i++) {
    // 每次跳过Key粒度包含元素的个数
    int q_smem_offset = ki + i * THREADS_PER_KEY * K_VEC_SIZE;
    q[i] = *reinterpret_cast<const K_vec *>(&q_smem[q_smem_offset]);
  }

  T *k_cache = &params.cache_kv[kv_bhi * max_seq_length * Dh + ki];
  T *k_cache_batch = &params.cache_kv[bbhi * max_seq_length * Dh + ki];
  const int *beam_offsets = params.beam_cache_offset
                                ? &params.beam_cache_offset[bi_seq_len_offset]
                                : nullptr;
  constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;
  constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;
  // 保证warpLevel执行相同
  int ti_end = div_up(end_seq - start_seq, K_PER_WARP) * K_PER_WARP + start_seq;
#pragma unroll
  for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
    const int beam_offset =
        beam_offsets ? beam_offsets[ti] * num_head * max_seq_length * Dh : 0;
    K_vec k[K_VECS_PER_THREAD];
    K_vec k_vec_zero;
    zero(k_vec_zero);
#pragma unroll
    for (int ii = 0; ii < K_VECS_PER_THREAD; ii++) {  // 这里其实就是列数
      int jj = ii * max_seq_length + ti;  // 每一行的Key粒度线程分布
      if (ti < end_seq) {
        if (beam_offset) {
          k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * max_seq_length)
                      ? *reinterpret_cast<const K_vec *>(
                            &k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B])
                      : k_vec_zero;
        } else {
          k[ii] = (Dh == Dh_MAX || jj * QK_ELTS_IN_16B < Dh * max_seq_length)
                      ? *reinterpret_cast<const K_vec *>(
                            &k_cache[jj * QK_ELTS_IN_16B])
                      : k_vec_zero;
        }
      }
    }
    /// Q dot cacheK with dh/mask（417行-436行）
    float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q, k, params.inv_sqrt_dh);

    if (ti < end_seq && tid % THREADS_PER_KEY == 0) {
      auto mask_bhi = params.mask_broadcast_num_heads ? bi : bhi;
      if (params.attn_mask) {
        T mask = params.attn_mask[mask_bhi * params.mask_length + ti];
        qk += static_cast<float>(mask);
      }
      qk_max = fmaxf(qk_max, qk);
      qk_smem[ti - start_seq] = qk;
    }
  }

/// reduce/softmax（438行-477行）
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
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
  for (int ti = tid; ti < end_seq - start_seq; ti += THREADS_PER_BLOCK) {
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }
  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  if (tid == 0) {
    float2 sum_max = {sum, qk_max};
    *reinterpret_cast<float2 *>(
        &params.qk_sum_max_split_seq[(bhsi + split_index) * 2]) = sum_max;
  }

  float inv_sum = __fdividef(1.f, sum + 1.e-6f);
  for (int ti = tid; ti < end_seq - start_seq; ti += THREADS_PER_BLOCK) {
    convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
  }
  __syncthreads();

  /// 读取cacheV && logits dot cacheV（479行-532行）
  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;  // V_vec就是16B
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif
  int vo = tid / THREADS_PER_VALUE + start_seq;
  int vi = (tid % THREADS_PER_VALUE) * V_VEC_SIZE;
  int v_cache_offset =
      params.cache_batch_size * kv_num_head * max_seq_length * Dh +
      kv_bhi * max_seq_length * Dh + vi;
  T *v_cache = &params.cache_kv[v_cache_offset];
  int v_cache_batch_offset =
      params.batch_size * num_head * max_seq_length * Dh +
      bbhi * max_seq_length * Dh + vi;
  T *v_cache_batch = &params.cache_kv[v_cache_batch_offset];
  V_vec_acum out;
  zero(out);
  constexpr int V_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
  if (Dh == Dh_MAX || vi < Dh) {
#pragma unroll
    for (int ti = vo; ti < end_seq; ti += V_PER_ITER) {
      V_vec v;
      const int beam_offset =
          beam_offsets ? beam_offsets[ti] * num_head * max_seq_length * Dh : 0;
      if (beam_offset) {
        v = *reinterpret_cast<const V_vec *>(
            &v_cache_batch[beam_offset + ti * Dh]);
      } else {
        v = *reinterpret_cast<const V_vec *>(&v_cache[ti * Dh]);
      }
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
      float logit = logits_smem[ti - start_seq];
      out = fma(logit, cast_to_float(v), out);
#else
      DataType_ logit = static_cast<DataType_>(logits_smem[ti - start_seq]);
      out = fma(logit, v, out);
#endif
    }
  }
  __syncthreads();

  /// reduce并把结果存回去
  if (Dh == Dh_MAX || vi < Dh) {
    vo -= start_seq;
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

    if (vo == 0) {
      // 这里还需考虑 params.cum_offsets
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      *(reinterpret_cast<V_vec_acum *>(
          &params.split_out[(bhsi + split_index) * Dh + vi])) = out;
#else
      *(reinterpret_cast<V_vec_acum_fp32_<V_vec>::Type *>(
          &params.split_out[(bhsi + split_index) * Dh + vi])) =
          cast_to_float(out);
#endif
    }
  }

#else   // CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
  assert(false);
#endif  // CUDA_ARCH_FP16_SUPPORTED(__CUDA_ARCH__)
}

// v3的每个线程块都启动了更多的线程（4x/8x）来处理长seq导致的多循环延迟
// 如10240Seq / 128StepsPerBlock = 80循环。用4x倍线程就是20个循环
template <typename T,
          int Dh,
          int Dh_MAX,
          int SPLTS_PER_BLOCK,
          typename StoreFunc>
__global__ void post_process_kernel_kai_v3(
    Masked_multihead_attention_params<T> params, StoreFunc store_func) {
  const int bi = blockIdx.y;
  const int tid = threadIdx.x;
  const int hi = blockIdx.x;
  const int bhi = (bi * params.num_head + hi);
  const int bhsi = (bi * params.num_head + hi) * params.split_seq;

  extern __shared__ float2 qk_sum_max_smem[];
  float max = -FLT_MAX;
  const int WARP_SIZE = 32;
  int WARPS_PER_BLOCK = blockDim.x / WARP_SIZE;

  for (int i = tid; i < params.real_split_each_batch[bi]; i += blockDim.x) {
    float2 sum_max = *reinterpret_cast<float2 *>(
        &params.qk_sum_max_split_seq[(bhsi + i) * 2]);
    max = fmaxf(sum_max.y, max);
    qk_sum_max_smem[i] = sum_max;
  }
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max = fmaxf(max, __shfl_xor_sync(uint32_t(-1), max, mask));
  }

  __shared__ float max_smem[32];
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  if (lane == 0) {
    max_smem[warp] = max;
  }
  __syncthreads();

  max = lane < WARPS_PER_BLOCK ? max_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    max = fmaxf(max, __shfl_xor_sync(uint32_t(-1), max, mask));
  }

  max = __shfl_sync(uint32_t(-1), max, 0);

  __shared__ float redu_smem[SPLTS_PER_BLOCK / 2][2][Dh_MAX];
  float sum = 0;
  float v = 0;
  int split_group_idx = tid / Dh_MAX;
  if ((tid % Dh_MAX) < Dh) {
#pragma unroll
    for (int i = split_group_idx; i < params.real_split_each_batch[bi];
         i += SPLTS_PER_BLOCK) {
      float2 sum_max = qk_sum_max_smem[i];
      float this_v = params.split_out[(bhsi + i) * Dh + (tid % Dh_MAX)];

      float real_this_sum = sum_max.x * __expf(sum_max.y - max);
      v += real_this_sum * this_v;
      sum += real_this_sum;
    }
    // 多个splits合并成一个
    for (int active_groups = SPLTS_PER_BLOCK; active_groups >= 2;
         active_groups /= 2) {
      int midpoint = active_groups / 2;
      if (split_group_idx >= midpoint && split_group_idx < active_groups) {
        redu_smem[split_group_idx - midpoint][0][tid % Dh_MAX] = v;
        redu_smem[split_group_idx - midpoint][1][tid % Dh_MAX] = sum;
      }
      __syncthreads();
      if (split_group_idx < midpoint) {
        v += redu_smem[split_group_idx][0][tid % Dh_MAX];
        sum += redu_smem[split_group_idx][1][tid % Dh_MAX];
      }
      __syncthreads();
    }
    if (split_group_idx == 0) {
      v /= sum;
      T tmp_v = (T)v;
      store_func.store(tmp_v, bhi * Dh + tid);
    }
  }
}

template <typename T>
inline size_t smem_size_in_bytes(
    const Masked_multihead_attention_params<T> &params,
    int dim_head,
    int threads_per_value,
    int threads_per_block) {
  size_t qk_sz = div_up(steps_per_block + 1, 4) * 16;
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

/// TODO: 在各种情况下测一测是不是真没问题
template <typename T>
inline size_t smem_size_in_bytes_kai(
    const Masked_multihead_attention_params<T> &params,
    int dim_head,
    int threads_per_value,
    int threads_per_block) {
  // for qk_smem and logits_smem(both float)
  size_t qk_sz = div_up(steps_per_block, 4) * 16;
  // for reduce (logits dot V) result
  int rows_per_red = threads_per_block / threads_per_value;
  size_t red_sz = rows_per_red * dim_head * sizeof(T) / 2;
  return max(qk_sz, red_sz);
}

#define MMHA_LAUNCH_KERNEL(T,                                                \
                           Dh,                                               \
                           Dh_MAX,                                           \
                           THDS_PER_KEY,                                     \
                           THDS_PER_VALUE,                                   \
                           THDS_PER_BLOCK,                                   \
                           stream,                                           \
                           load_func)                                        \
  constexpr auto preprocess_fn = preprocess<T,                               \
                                            Dh,                              \
                                            Dh_MAX,                          \
                                            THDS_PER_VALUE,                  \
                                            THDS_PER_BLOCK,                  \
                                            decltype(load_func)>;            \
  constexpr auto kernel_fn =                                                 \
      masked_multihead_attention_kernel<T,                                   \
                                        Dh,                                  \
                                        Dh_MAX,                              \
                                        THDS_PER_KEY,                        \
                                        THDS_PER_VALUE,                      \
                                        THDS_PER_BLOCK,                      \
                                        decltype(load_func)>;                \
  dim3 grid(params.num_head, params.batch_size);                             \
  preprocess_fn<<<grid, Dh_MAX / 2, 0, stream>>>(preprocess_params,          \
                                                 load_func);                 \
  size_t smem_sz =                                                           \
      smem_size_in_bytes_kai<T>(params, Dh, THDS_PER_VALUE, THDS_PER_BLOCK); \
  if (smem_sz > 0xc000) {                                                    \
    cudaFuncSetAttribute(                                                    \
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);    \
  }                                                                          \
  grid = {params.split_seq, params.num_head, params.batch_size};             \
  kernel_fn<<<grid, THDS_PER_BLOCK, smem_sz, stream>>>(params, load_func)

template <typename T, int Dh, int Dh_MAX, typename LoadFunc>
void fmha_launch_kernel(const preprocess_params<T> preprocess_params,
                        const Masked_multihead_attention_params<T> &params,
                        const cudaStream_t &stream,
                        LoadFunc load_func) {
  constexpr int THREADS_PER_VALUE = Dh_MAX * sizeof(T) / 16;
  if (params.timestep < 64) {
    MMHA_LAUNCH_KERNEL(
        T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 64, stream, load_func);
  } else {
    // 因为每个block处理的seq长度固定了，所以最佳参数也可以固定。
    MMHA_LAUNCH_KERNEL(
        T, Dh, Dh_MAX, 4, THREADS_PER_VALUE, 128, stream, load_func);
  }
}

template <typename T, typename LoadFunc, typename StoreFunc_kai>
void fmha_impl(const phi::GPUContext &dev_ctx,
               const preprocess_params<T> preprocess_params,
               const Masked_multihead_attention_params<T> &params,
               int dim_head,
               LoadFunc load_func,
               StoreFunc_kai store_func_kai) {
  switch (dim_head) {
    case 10:
      fmha_launch_kernel<T, 10, 32>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 16:
      fmha_launch_kernel<T, 16, 32>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 26:
      fmha_launch_kernel<T, 26, 32>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 32:
      fmha_launch_kernel<T, 32, 32>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 64:
      fmha_launch_kernel<T, 64, 64>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    // for opt model
    case 80:
      fmha_launch_kernel<T, 80, 128>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 96:
      fmha_launch_kernel<T, 96, 128>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 128:
      fmha_launch_kernel<T, 128, 128>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    case 192:
      fmha_launch_kernel<T, 192, 256>(
          preprocess_params, params, dev_ctx.stream(), load_func);
      break;
    default:
      PADDLE_THROW(
          phi::errors::Unimplemented("Dim_head = %d is unsupport!", dim_head));
  }

  /// v3   BLK_SIZE = Dh_MAX * 4 以应对长seq的大量循环
  dim3 grid(params.num_head, params.batch_size);
  int smem_sz = params.split_seq * sizeof(float2);
  switch (dim_head) {
    case 10:
      post_process_kernel_kai_v3<T, 10, 32, 8, StoreFunc_kai>
          <<<grid, 256, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 16:
      post_process_kernel_kai_v3<T, 16, 32, 8, StoreFunc_kai>
          <<<grid, 256, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 26:
      post_process_kernel_kai_v3<T, 26, 32, 8, StoreFunc_kai>
          <<<grid, 256, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 32:
      post_process_kernel_kai_v3<T, 32, 32, 8, StoreFunc_kai>
          <<<grid, 256, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 64:
      post_process_kernel_kai_v3<T, 64, 64, 4, StoreFunc_kai>
          <<<grid, 256, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 80:
      post_process_kernel_kai_v3<T, 80, 128, 4, StoreFunc_kai>
          <<<grid, 512, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 96:
      post_process_kernel_kai_v3<T, 96, 128, 4, StoreFunc_kai>
          <<<grid, 512, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 128:
      post_process_kernel_kai_v3<T, 128, 128, 4, StoreFunc_kai>
          <<<grid, 512, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 192:
      post_process_kernel_kai_v3<T, 192, 256, 4, StoreFunc_kai>
          <<<grid, 1024, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    case 256:
      post_process_kernel_kai_v3<T, 256, 256, 4, StoreFunc_kai>
          <<<grid, 1024, smem_sz, dev_ctx.stream()>>>(params, store_func_kai);
      break;
    default:
      PADDLE_THROW(
          phi::errors::Unimplemented("Dim_head = %d is unsupport!", dim_head));
  }
}

template <typename T>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const preprocess_params<T> preprocess_params,
                  const Masked_multihead_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const phi::DenseTensor *dequant_qkv_scales = nullptr,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore_kai<T, int8_t> store_func_kai(out_tensor->data<int8_t>(),
                                            quant_round_type,
                                            quant_fmha_out_scale,
                                            quant_max_bound,
                                            quant_min_bound);
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore_kai<T, int8_t> store_func_kai(out_tensor->data<int8_t>(),
                                            quant_round_type,
                                            quant_fmha_out_scale,
                                            quant_max_bound,
                                            quant_min_bound);
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore_kai<T> store_func_kai(out_tensor->data<T>());
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore_kai<T> store_func_kai(out_tensor->data<T>());
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  }
}

template <typename T>
void DispatchFMHA(const phi::GPUContext &dev_ctx,
                  const phi::DenseTensor &qkv_tensor,
                  const phi::DenseTensor &shift,
                  const phi::DenseTensor &smooth,
                  const preprocess_params<T> preprocess_params,
                  const Masked_multihead_attention_params<T> &params,
                  int num_head,
                  int dim_head,
                  phi::DenseTensor *out_tensor,
                  const phi::DenseTensor *dequant_qkv_scales = nullptr,
                  const float quant_fmha_out_scale = -1,
                  const int quant_round_type = 1,
                  const float quant_max_bound = 127.0f,
                  const float quant_min_bound = -127.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore_kai<T, int8_t, true> store_func_kai(out_tensor->data<int8_t>(),
                                                  shift.data<T>(),
                                                  smooth.data<T>(),
                                                  num_head * dim_head,
                                                  quant_round_type,
                                                  quant_fmha_out_scale,
                                                  quant_max_bound,
                                                  quant_min_bound);
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore_kai<T, int8_t, true> store_func_kai(out_tensor->data<int8_t>(),
                                                  shift.data<T>(),
                                                  smooth.data<T>(),
                                                  num_head * dim_head,
                                                  quant_round_type,
                                                  quant_fmha_out_scale,
                                                  quant_max_bound,
                                                  quant_min_bound);
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * num_head * dim_head);
    MMHAStore_kai<T, T, true> store_func_kai(out_tensor->data<T>(),
                                             shift.data<T>(),
                                             smooth.data<T>(),
                                             num_head * dim_head);
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore_kai<T, T, true> store_func_kai(out_tensor->data<T>(),
                                             shift.data<T>(),
                                             smooth.data<T>(),
                                             num_head * dim_head);
    fmha_impl(dev_ctx,
              preprocess_params,
              params,
              dim_head,
              load_func,
              store_func_kai);
  }
}

struct NormalVersion {};
struct UnusedVersion {};

template <typename T>
struct DispatchDtypeTrait {
  using FuncVersion = NormalVersion;
};

template <>
struct DispatchDtypeTrait<int32_t> {
  using FuncVersion = UnusedVersion;
};

template <typename T, typename Context>
void DispatchWithDtype(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &cache_kv,
                       const paddle::optional<DenseTensor> &bias,
                       const paddle::optional<DenseTensor> &src_mask,
                       const paddle::optional<DenseTensor> &cum_offsets,
                       const paddle::optional<DenseTensor> &sequence_lengths,
                       const paddle::optional<DenseTensor> &rotary_tensor,
                       const paddle::optional<DenseTensor> &beam_cache_offset,
                       const paddle::optional<DenseTensor> &qkv_out_scale,
                       const paddle::optional<DenseTensor> &out_shift,
                       const paddle::optional<DenseTensor> &out_smooth,
                       int seq_len,
                       int rotary_emb_dims,
                       const bool use_neox_rotary_style,
                       const float out_scale,
                       const int quant_round_type,
                       const float quant_max_bound,
                       const float quant_min_bound,
                       DenseTensor *out,
                       DenseTensor *cache_kv_out,
                       DenseTensor *beam_cache_offset_out,
                       NormalVersion) {
  const auto &x_dims = x.dims();
  int bsz = x_dims[0];
  int cache_bsz = cache_kv.dims()[1];
  int max_seq_len = cache_kv.dims()[3];
  int dim_head = cache_kv.dims()[4];
  int timestep = max_seq_len;
  float inv_sqrt_dh = 1. / sqrt(dim_head);

  int k_num_head = cache_kv.dims()[2];
  int v_num_head = k_num_head;
  // this num_head means query's head
  int num_head =
      x.dims()[x.dims().size() - 1] / dim_head - k_num_head - v_num_head;

  Masked_multihead_attention_params<T> mmha_params;
  preprocess_params<T> preprocess_params;
  bool mask_broadcast_num_heads = true;

  preprocess_params.qkv = const_cast<T *>(x.data<T>());  // 前处理强行覆盖当前Q

  mmha_params.add_qkv_bias = false;
  preprocess_params.add_qkv_bias = false;
  if (bias) {
    mmha_params.add_qkv_bias = true;
    mmha_params.qkv_bias = const_cast<T *>(bias->data<T>());
    preprocess_params.add_qkv_bias = true;
    preprocess_params.qkv_bias = const_cast<T *>(bias->data<T>());
  }

  if (src_mask) {
    if (src_mask->dims()[1] == 1) {
      mask_broadcast_num_heads = true;
    } else if (src_mask->dims()[1] == num_head) {
      mask_broadcast_num_heads = false;
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Unknow dimension for attn_mask, the num_head(2nd) "
          "dimension is invalid, it should be 1 or num_head(%d), "
          "but got %d",
          num_head,
          src_mask->dims()[1]));
    }
    mmha_params.attn_mask = src_mask->data<T>();
    mmha_params.mask_length = src_mask->dims()[3];
    timestep = src_mask->dims()[3] - 1;
  }

  if (out_scale > 0) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  if (sequence_lengths) {
    mmha_params.sequence_lengths = sequence_lengths->data<int>();
    preprocess_params.sequence_lengths = sequence_lengths->data<int>();
  }

  if (cum_offsets) {
    mmha_params.cum_offsets = cum_offsets->data<int>();
  } else {
    mmha_params.cum_offsets = nullptr;
  }

  if (rotary_emb_dims > 0) {
    preprocess_params.rotary_emb = rotary_tensor->data<float>();
    mmha_params.rotary_emb = rotary_tensor->data<float>();
  } else {
    preprocess_params.rotary_emb = nullptr;
    mmha_params.rotary_emb = nullptr;
  }

  if (beam_cache_offset) {
    mmha_params.beam_cache_offset = beam_cache_offset->data<int>();
    mmha_params.beam_width = beam_cache_offset->dims()[1];
  }

  mmha_params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  mmha_params.cache_kv = const_cast<T *>(cache_kv_out->data<T>());
  mmha_params.neox_rotary_style = use_neox_rotary_style;
  mmha_params.batch_size = bsz;
  mmha_params.cache_batch_size = cache_bsz;
  mmha_params.num_head = num_head;
  mmha_params.kv_num_head = k_num_head;
  mmha_params.timestep = timestep;
  mmha_params.seq_len = seq_len;
  mmha_params.max_seq_length = max_seq_len;
  mmha_params.inv_sqrt_dh = inv_sqrt_dh;
  mmha_params.rotary_emb_dims = rotary_emb_dims;

  int split_seq = timestep / steps_per_block + 1;
  mmha_params.split_seq = split_seq;  // 我们为一个seq启动split_seq个线程块

  phi::DenseTensor qk_sum_max_split_seq;
  qk_sum_max_split_seq.Resize({{bsz, num_head, split_seq, 2}});
  dev_ctx.template Alloc<float>(&qk_sum_max_split_seq,
                                qk_sum_max_split_seq.numel() * sizeof(float));
  mmha_params.qk_sum_max_split_seq = qk_sum_max_split_seq.data<float>();

  phi::DenseTensor real_split_each_batch;
  real_split_each_batch.Resize({{bsz}});
  dev_ctx.template Alloc<int>(&real_split_each_batch,
                              real_split_each_batch.numel() * sizeof(int));
  mmha_params.real_split_each_batch = real_split_each_batch.data<int>();

  phi::DenseTensor split_out;
  split_out.Resize({{bsz, num_head, split_seq, dim_head}});
  dev_ctx.template Alloc<float>(&split_out, split_out.numel() * sizeof(float));
  mmha_params.split_out = split_out.data<float>();

  preprocess_params.kv_num_head = k_num_head;
  preprocess_params.num_head = num_head;
  preprocess_params.max_seq_length = max_seq_len;
  preprocess_params.timestep = timestep;
  preprocess_params.neox_rotary_style = use_neox_rotary_style;
  preprocess_params.rotary_emb_dims = rotary_emb_dims;
  preprocess_params.batch_size = bsz;
  preprocess_params.cache_kv = const_cast<T *>(cache_kv_out->data<T>());
  preprocess_params.cache_batch_size = cache_bsz;

  if (out_shift) {
    DispatchFMHA<T>(dev_ctx,
                    x,
                    *(out_shift.get_ptr()),
                    *(out_smooth.get_ptr()),
                    preprocess_params,
                    mmha_params,
                    num_head,
                    dim_head,
                    out,
                    qkv_out_scale.get_ptr(),
                    out_scale,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound);
  } else {
    DispatchFMHA<T>(dev_ctx,
                    x,
                    preprocess_params,
                    mmha_params,
                    num_head,
                    dim_head,
                    out,
                    qkv_out_scale.get_ptr(),
                    out_scale,
                    quant_round_type,
                    quant_max_bound,
                    quant_min_bound);
  }
}

template <typename T, typename Context>
void DispatchWithDtype(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &cache_kv,
                       const paddle::optional<DenseTensor> &bias,
                       const paddle::optional<DenseTensor> &src_mask,
                       const paddle::optional<DenseTensor> &cum_offsets,
                       const paddle::optional<DenseTensor> &sequence_lengths,
                       const paddle::optional<DenseTensor> &rotary_tensor,
                       const paddle::optional<DenseTensor> &beam_cache_offset,
                       const paddle::optional<DenseTensor> &qkv_out_scale,
                       const paddle::optional<DenseTensor> &out_shift,
                       const paddle::optional<DenseTensor> &out_smooth,
                       int seq_len,
                       int rotary_emb_dims,
                       const bool use_neox_rotary_style,
                       const float out_scale,
                       const int quant_round_type,
                       const float quant_max_bound,
                       const float quant_min_bound,
                       DenseTensor *out,
                       DenseTensor *cache_kv_out,
                       DenseTensor *beam_cache_offset_out,
                       UnusedVersion) {}

#endif  // PADDLE_WITH_HIP

constexpr unsigned int str2int(const char *str, int h = 0) {
  return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

template <typename T, typename Context>
void MMHAKernel(const Context &dev_ctx,
                const DenseTensor &x,
                const DenseTensor &cache_kv,
                const paddle::optional<DenseTensor> &bias,
                const paddle::optional<DenseTensor> &src_mask,
                const paddle::optional<DenseTensor> &cum_offsets,
                const paddle::optional<DenseTensor> &sequence_lengths,
                const paddle::optional<DenseTensor> &rotary_tensor,
                const paddle::optional<DenseTensor> &beam_cache_offset,
                const paddle::optional<DenseTensor> &qkv_out_scale,
                const paddle::optional<DenseTensor> &out_shift,
                const paddle::optional<DenseTensor> &out_smooth,
                int seq_len,
                int rotary_emb_dims,
                const bool use_neox_rotary_style,
                const std::string &compute_dtype,
                const float out_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                DenseTensor *out,
                DenseTensor *cache_kv_out,
                DenseTensor *beam_cache_offset_out) {
#ifndef PADDLE_WITH_HIP
  if (x.dtype() == phi::DataType::INT32) {
    switch (str2int(compute_dtype.c_str())) {
      case str2int("fp16"):
        DispatchWithDtype<phi::dtype::float16, Context>(
            dev_ctx,
            x,
            cache_kv,
            bias,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out,
            cache_kv_out,
            beam_cache_offset_out,
            typename DispatchDtypeTrait<phi::dtype::float16>::FuncVersion{});
        break;
#if CUDA_VERSION >= 11000
      case str2int("bf16"):
        DispatchWithDtype<phi::dtype::bfloat16, Context>(
            dev_ctx,
            x,
            cache_kv,
            bias,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out,
            cache_kv_out,
            beam_cache_offset_out,
            typename DispatchDtypeTrait<phi::dtype::bfloat16>::FuncVersion{});
        break;
#endif
      case str2int("fp32"):
        DispatchWithDtype<float, Context>(
            dev_ctx,
            x,
            cache_kv,
            bias,
            src_mask,
            cum_offsets,
            sequence_lengths,
            rotary_tensor,
            beam_cache_offset,
            qkv_out_scale,
            out_shift,
            out_smooth,
            seq_len,
            rotary_emb_dims,
            use_neox_rotary_style,
            out_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
            out,
            cache_kv_out,
            beam_cache_offset_out,
            typename DispatchDtypeTrait<float>::FuncVersion{});
        break;
      default:
        PADDLE_THROW(phi::errors::InvalidArgument(
            "In the case of quantization enabled with Input(x) INT32, "
            "Attr(compute_dtype) must be set in (bf16, fp16, fp32), "
            "but get compute_dtype (%s)",
            compute_dtype));
    }
  } else {
    DispatchWithDtype<T, Context>(
        dev_ctx,
        x,
        cache_kv,
        bias,
        src_mask,
        cum_offsets,
        sequence_lengths,
        rotary_tensor,
        beam_cache_offset,
        qkv_out_scale,
        out_shift,
        out_smooth,
        seq_len,
        rotary_emb_dims,
        use_neox_rotary_style,
        out_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        out,
        cache_kv_out,
        beam_cache_offset_out,
        typename DispatchDtypeTrait<T>::FuncVersion{});
  }
#endif  // PADDLE_WITH_HIP
}

}  // namespace fusion
}  // namespace phi

#if CUDA_VERSION >= 11000
PD_REGISTER_KERNEL(masked_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMHAKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int32_t) {}
#else
PD_REGISTER_KERNEL(masked_multihead_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MMHAKernel,
                   float,
                   phi::dtype::float16,
                   int32_t) {}
#endif
