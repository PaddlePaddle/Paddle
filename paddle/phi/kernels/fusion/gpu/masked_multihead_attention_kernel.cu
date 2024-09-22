#include <assert.h>
#include <float.h>
#include <type_traits>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/fusion/gpu/mmha_util.cu.h"

/// Multi-block mmha kernel can only be selected when CUDA >= 11.7
/// 我们这里没有处理这个宏
#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

namespace phi {
namespace fusion {

#ifndef PADDLE_WITH_HIP

constexpr unsigned int str2int(const char *str, int h = 0) {
  return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

template <typename T>
struct Masked_multihead_attention_params {
  float *partial_out;
  float *partial_sum;
  float *partial_max;

  int *block_counter;
  // qkv_out, [B, 1(seq_len), 3, num_head * dim_head]
  const T *qkv;

  // bias, [3, num_head, dim_head]
  T *qkv_bias;

  // [2, B, num_head, max_seq_len, dim_head]
  // k [B, num_head, dim_head/chunk_eles, max_seq_len, chunk_eles]
  // v [B, num_head, max_seq_len, dim_head]
  T *cache_kv;

  // [B, max_seq_len]
  const int *beam_cache_offset = nullptr;

  const int *sequence_lengths{nullptr};

  // The RoPE embedding, [2, B, rotary_seq_len, 1, dim_head]
  // rotary_emb_dims = 1 if pos_ids_extra is null else 2
  const float *rotary_emb;

  // TODO(wangxi): optimize with input_lengths and max_input_len?
  // [bsz, 1, 1, time_step(cache_seq_length)+1]
  const T *attn_mask;

  int rotary_emb_dims;
  int batch_size;  // batch * beam
  int beam_width;
  int cache_batch_size;

  unsigned num_heads_kv;
  unsigned num_heads;
  unsigned timestep;
  mutable unsigned timesteps_per_block;
  mutable unsigned seq_len_tile;
  int dim_head;

  unsigned max_seq_len_tile;
  unsigned min_seq_len_tile;
  int dynamic_block_size;

  int max_seq_length;

  // 1.f / sqrt(Dh)
  float inv_sqrt_dh;

  bool add_qkv_bias;
  bool neox_rotary_style;
  bool mask_broadcast_num_heads;

  mutable bool multi_block_mode;
};

/// 因为加入了K_LOOP_UNROLL, 我(YKTian)把加载cacheK的思路转为一个thread做一个LDG.128
/// 加载cacheV的思路是一个thread做一个LDG.128
template <typename T,
          unsigned Dh,
          unsigned THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc,
          bool DO_MULTI_BLOCK = false,
          unsigned THREADS_PER_KEY = getThreadsPerKey<Dh>(),
          unsigned THREADS_PER_VALUE = getThreadsPerValue<T, Dh>(),
          unsigned K_LOOP_UNROLL = 4,
          unsigned V_LOOP_UNROLL = 8>
__global__ void masked_multihead_attention_kernel(
    Masked_multihead_attention_params<T> params,
    LoadFunc load_func,
    StoreFunc store_func) {
  //// 初始化变量（对应TRTLLM 1310-1505）
  // dim3 grid{params.seq_len_tile, params.num_heads, params.batch_size};
  const unsigned bi{blockIdx.z};
  const unsigned hi{blockIdx.y};
  // The column tile along L dimension on K^T -- noted as T_c in flash-attention
  // paper
  const unsigned c_tile{DO_MULTI_BLOCK ? blockIdx.x : 0};
  unsigned const tid{threadIdx.x};

  // The current timestep (including paddings).
  // It is only used to calculate the smem stride.
  unsigned const timestep =
      DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep;
  // The actual sequence length excluding the paddings.
  // 暂时同 tlength kv_loop_length
  const unsigned real_time_step = params.sequence_lengths == nullptr
                                      ? params.timestep
                                      : params.sequence_lengths[bi];
  auto const timesteps_per_block = params.timesteps_per_block;

  unsigned start_seq = 0;
  unsigned end_seq = real_time_step;
  bool is_last_block_static = (DO_MULTI_BLOCK == false);
  unsigned real_seq_len_tile = (real_time_step - 1) / timesteps_per_block + 1;

  if constexpr (DO_MULTI_BLOCK) {
    if (c_tile >= real_seq_len_tile) return;

    start_seq = c_tile * timesteps_per_block;
    end_seq = start_seq + timesteps_per_block;
    if (c_tile == real_seq_len_tile - 1) {
      is_last_block_static = true;
      end_seq = real_time_step;
    }
  }
  const int curr_seq_section = end_seq - start_seq;

  constexpr unsigned WARP_SIZE{32};
  constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};
  constexpr unsigned Dh_MAX = getDhMax(Dh);
  constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
  static_assert(Dh_MAX >= WARP_SIZE);
  static_assert(Dh_MAX >= Dh);
  // THREADS_PER_KEY in {1, 2, 4}
  static_assert(THREADS_PER_KEY <= 4);
  static_assert(Dh_MAX % THREADS_PER_KEY == 0);

  extern __shared__ char smem_[];
  auto qk_smem = reinterpret_cast<float *>(smem_);

  char *logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
  if (sizeof(T) != 4) {
    logits_smem_ += divUp(timestep + 1, 4u) * 16;
  }
  T *logits_smem = reinterpret_cast<T *>(logits_smem_);
#else
  float *logits_smem = reinterpret_cast<float *>(logits_smem_);
#endif

  T *out_smem = reinterpret_cast<T *>(smem_);

  __shared__ float red_smem[WARPS_PER_BLOCK * 2];

  constexpr unsigned K_VEC_SIZE = 16u / sizeof(T);
  static_assert(Dh_MAX % K_VEC_SIZE == 0);

  using Qk_vec = typename Qk_vec_<T, Dh_MAX>::Type;

  __shared__ __align__(sizeof(Qk_vec)) T q_smem[Dh_MAX];

  constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec) / sizeof(T)};
  static_assert(Dh_MAX % QK_VEC_SIZE == 0);
  constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
  static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

  // The head index of keys and values adjusted for MQA/GQA.
  unsigned const qhead_per_kv{params.num_heads / params.num_heads_kv};
  unsigned const hi_kv{hi / qhead_per_kv};
  // The number of heads.
  unsigned const num_heads = params.num_heads;
  // The number of heads for keys and values adjusted for MQA/GQA.
  unsigned const num_heads_kv = params.num_heads_kv;

  // qkv [B, S=1, num_head + 2 * num_heads_kv, head_dim]
  // this hi means the head index in query!
  unsigned qkv_bi = bi * (num_heads + 2 * num_heads_kv) * Dh;

  const unsigned bhi = bi * num_heads + hi;
  const unsigned bhi_kv = bi * num_heads_kv + hi_kv;

  float qk_max = -FLT_MAX;
  float qk = 0.0F;

  auto const qk_vec_idx = tid * QK_VEC_SIZE;
  auto const is_valid_qk_vec = qk_vec_idx < Dh;
  //// 加载当前q k 并作位置编码 +bias（1505-1729）
  Qk_vec q, k;
  if (is_valid_qk_vec) {
    auto const q_offset = qkv_bi + hi * Dh + qk_vec_idx;
    load_func.template load<Qk_vec>(q, q_offset);
    // q = *reinterpret_cast<Qk_vec const *>(&params.q[q_offset]);

    if (is_last_block_static) {
      auto const k_offset = qkv_bi + (num_heads + hi_kv) * Dh + qk_vec_idx;
      load_func.template load<Qk_vec>(k, k_offset);
      // k = *reinterpret_cast<Qk_vec const *>(&params.k[k_offset]);
    }
  }
  /// TODO:省略了位置编码 +bias

  //// 将当前q 存入共享内存中, k存入全局内存中。计算当前q dot
  /// k（1733-1769-1839）
  Qk_vec zero_q;
  zero(zero_q);
  if (qk_vec_idx < Dh_MAX) {
    reinterpret_cast<Qk_vec *>(&q_smem[qk_vec_idx])[0] =
        is_valid_qk_vec ? q : zero_q;
  }

  if (is_last_block_static) {
    if (qk_vec_idx < Dh_MAX) {
      qk = dot<Qk_vec, Qk_vec>(q, k);
      if (QK_VECS_PER_Dh_MAX <= WARP_SIZE) {
#pragma unroll
        for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2) {
          qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk, mask);
        }
      }
    }
    if (QK_VECS_PER_Dh_MAX > WARP_SIZE) {
      constexpr int WARPS_PER_RED =
          (QK_VECS_PER_Dh_MAX + WARP_SIZE - 1) / WARP_SIZE;
      qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    if (tid == 0) {
      qk *= params.inv_sqrt_dh;
      qk_max = qk;
      ///TODO: attn_mask
      qk_smem[real_time_step - start_seq] = qk;
    }
  }

  __syncthreads();

  //// q dot cacheK （1841-1961-2216）
  /// cacheK [B, num_head, dim_head/chunk_eles, max_seq_len, chunk_eles]
  using K_vec = typename V_vec_<T, K_VEC_SIZE>::Type;  // 16B
  static_assert(Dh_MAX % K_VEC_SIZE == 0);
  const bool is_leader = trtllm_Qk_dot<T, THREADS_PER_KEY>::is_leader(tid);

  /// 一个chunk是 处理同一个key的多个线程 一次取的元素的集合
  constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};
  static_assert(Dh % K_ELTS_PER_CHUNK == 0);
  // {chunk_id, inner_chunk_id(elt_id)}
  // {tid/THREADS_PER_KEY, (tid % THREADS_PER_KEY) * K_VEC_SIZE}
  auto const k_idx = chunk_index<T, K_vec, THREADS_PER_KEY>(tid);

  constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
  
  static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

  K_vec q_vec[K_VECS_PER_THREAD];
#pragma unroll
  for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
    q_vec[ii] = *reinterpret_cast<K_vec const *>(
        &q_smem[ii * K_ELTS_PER_CHUNK + k_idx.y]);
  }

  constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
  constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
  constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
  constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

  /// batch as many LDGs as possible.
  // Pick a number of keys to make sure all the threads of a warp enter (due to
  // shfl_sync).
  auto const ti_end =
      div_up(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

  // 从这个指针开始 加载 [dim_head/chunk_eles, max_seq_len, chunk_eles]
  //                  [k_vec_i, start_seq + k_loop * K_PER_ITER + k_idx.x,
  //                  k_idx.y]
  T *k_cache = &params.cache_kv[bhi_kv * params.max_seq_length * Dh];

  for (int ti = k_idx.x; ti < ti_end; ti += UNROLLED_K_PER_ITER) {
    const int time_now = start_seq + ti;
    K_vec k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];
#pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop) {
#pragma unroll
      for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i) {
        /// Make sure we read data within the bound.
        // Dh OOB values will be handled by zero_q.
        /// int dh_offs = min(Dh - K_VEC_SIZE, k_vec_i * K_ELTS_PER_CHUNK +
        /// k_idx.y);
        const int local_idx0 = min(Dh / K_ELTS_PER_CHUNK, k_vec_i);
        const int local_idx2 =
            k_vec_i > local_idx0 ? (THREADS_PER_KEY - 1) * K_VEC_SIZE : k_idx.y;

        // Seq OOB values will be masked out when storing back to smem.
        const int local_idx1 =
            min(end_seq - 1, time_now + k_loop * K_PER_ITER);

        // (k_vec_i * params.max_seq_length + k_idx.x + start_seq + k_loop * K_PER_ITER) * K_VEC_SIZE + k_idx.y;
        const int cache_k_local_off =
            (local_idx0 * params.max_seq_length + local_idx1) *
                K_ELTS_PER_CHUNK +
            local_idx2;

        k_vec_cache[k_loop][k_vec_i] =
            *reinterpret_cast<K_vec const *>(&k_cache[cache_k_local_off]);
      }
    }

#pragma unroll
    for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop) {
      int const local_ti = ti + k_loop * K_PER_ITER;
      int const local_time_now = local_ti + start_seq;

      // Perform the dot product and normalize qk.
      // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
      K_vec k_vec[K_VECS_PER_THREAD];
#pragma unroll
      for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i) {
        k_vec[k_vec_i] =
            *reinterpret_cast<K_vec *>(&k_vec_cache[k_loop][k_vec_i]);
      }

      float qk_ = trtllm_Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) *
                  params.inv_sqrt_dh;

      // For multi-block mode, we need to make sure it will not be OOB.
      if (local_time_now < end_seq && is_leader) {
        // Calculate the max for softmax.
        qk_max = fmaxf(qk_max, qk_);
        // Store the product to shared memory.
        qk_smem[local_ti] = qk_;
      }
    }
  }

  //// softmax （2218-2376）
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
  // Leader threads will be in the dignonal when using HMMA.
  if (THREADS_PER_KEY <= 4) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 4));
  }
  if (THREADS_PER_KEY <= 8) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 9));
  }
  if (THREADS_PER_KEY <= 16) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 18));
  }
#else
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
#endif  // defined MMHA_USE_HMMA

  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;

  if (lane == 0) {
    red_smem[warp] = qk_max;
  }

  __syncthreads();

  // cache_k [B, num_head, dim_head/chunk_eles, max_seq_len, chunk_eles]
  // Get the c_tile_id that handles the current timestep.
  const int QK_VECS_PER_CHUNK = K_ELTS_PER_CHUNK / QK_VEC_SIZE;
  // {tid/QK_VECS_PER_CHUNK, (tid % QK_VECS_PER_CHUNK) * QK_VEC_SIZE}
  auto const curr_k_idx = chunk_index<T, Qk_vec, QK_VECS_PER_CHUNK>(tid);
  const int curr_ko = curr_k_idx.x;  // tid / QK_VECS_PER_CHUNK;
  const int curr_ki = curr_k_idx.y;  // (tid % QK_VECS_PER_CHUNK) * QK_VEC_SIZE;

  if (is_last_block_static && (hi == hi_kv * qhead_per_kv) && is_valid_qk_vec) {
    int curr_k_off = bhi_kv * params.max_seq_length * Dh +
                     curr_ko * params.max_seq_length * K_ELTS_PER_CHUNK +
                     real_time_step * K_ELTS_PER_CHUNK + curr_ki;
    *reinterpret_cast<Qk_vec *>(&params.cache_kv[curr_k_off]) = k;
  }

  qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }

  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  float sum = 0.f;
  const int logit_loop_end =
      is_last_block_static ? curr_seq_section : curr_seq_section - 1;
  for (int ti = tid; ti <= logit_loop_end; ti += THREADS_PER_BLOCK) {
    float logit = __expf(qk_smem[ti] - qk_max);
    sum += logit;
    qk_smem[ti] = logit;
  }
  sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

  float inv_sum = __fdividef(1.f, sum + 1.e-6f);

  for (int ti = tid; ti <= logit_loop_end; ti += THREADS_PER_BLOCK) {
    if (!DO_MULTI_BLOCK)
    {
      convert_from_float(logits_smem[ti], qk_smem[ti] * inv_sum);
    }
    else
    {
      convert_from_float(logits_smem[ti], qk_smem[ti]);
    }
  }

  __syncthreads();

  //// logits dot cacheV  （2381-2527 + 2617-2644）
  static_assert(Dh_MAX % THREADS_PER_VALUE == 0);
  constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
  using V_vec = typename V_vec_<T, V_VEC_SIZE>::Type;
  static_assert(V_VEC_SIZE == sizeof(V_vec) / sizeof(T));
  // {tid/THREADS_PER_VALUE, (tid % THREADS_PER_VALUE) * V_VEC_SIZE}
  auto const v_idx = chunk_index<T, V_vec, THREADS_PER_VALUE>(tid);
  // The value computed by this thread.
  auto const vo = v_idx.x;
  // The hidden dimensions computed by this particular thread.
  auto const vi = v_idx.y;

  constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
  // The number of unrolled keys per ieration.
  constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

  bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

  // v_cache [B, num_head, max_seq_len, dim_head]
  T *v_cache =
      &params.cache_kv[(params.cache_batch_size * num_heads_kv + bhi_kv) *
                       params.max_seq_length * Dh];

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
  using V_vec_acum = typename V_vec_acum_fp32_<V_vec>::Type;
#else
  using V_vec_acum = V_vec;
#endif
  V_vec_acum out;
  zero(out);

  if (is_valid_vi) {
    for (int ti = vo; ti < curr_seq_section; ti += UNROLLED_V_PER_ITER) {
      V_vec v_vec_cache[V_LOOP_UNROLL];
#pragma unroll
      for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++) {
        const int local_time_now = ti + (v_loop * V_PER_ITER);
        const int time_now =
            min(start_seq + local_time_now, curr_seq_section - 1);
        v_vec_cache[v_loop] =
            *reinterpret_cast<V_vec const *>(&v_cache[time_now * Dh + vi]);
      }

      for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++) {
        const int local_time_now = ti + (v_loop * V_PER_ITER);
        const int time_now = start_seq + local_time_now;
        V_vec v_vec = *reinterpret_cast<V_vec *>(&v_vec_cache[v_loop]);

        bool const is_mask = DO_MULTI_BLOCK && time_now >= end_seq;
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        float logit =
            is_mask ? 0.f
                    : *reinterpret_cast<float *>(logits_smem + local_time_now);
        out = fma(logit, cast_to_float(v_vec), out);
#else   // MMHA_USE_FP32_ACUM_FOR_LOGITS
        T logit = is_mask ? T(0.f) : *(logits_smem + local_time_now);
        out = fma(logit, v_vec, out);
#endif  // MMHA_USE_FP32_ACUM_FOR_LOGITS
      }
    }
  }
  __syncthreads();

  //// 当前 logits dot v 和当前v写回全局内存(2530-2617)
  const int v_offset_base = qkv_bi + (num_heads + num_heads_kv + hi_kv) * Dh;
  if (is_last_block_static && (vo == real_time_step % V_PER_ITER) &&
      is_valid_vi) {
    const int v_offset = v_offset_base + vi;
    V_vec v = *reinterpret_cast<V_vec const *>(&params.qkv[v_offset]);
    if(hi == hi_kv * qhead_per_kv){
      *reinterpret_cast<V_vec *>(&v_cache[real_time_step * Dh + vi]) = v;
    }

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
    out = fma(logits_smem[real_time_step - start_seq], cast_to_float(v), out);
#else   // MMHA_USE_FP32_ACUM_FOR_LOGITS
    out = fma(logits_smem[real_time_step - start_seq], v, out);
#endif  // MMHA_USE_FP32_ACUM_FOR_LOGITS
  }
  __syncthreads();

#pragma unroll
  for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2) {
    // The midpoint in the number of active groups.
    int midpoint = active_groups / 2;

    // The upper part of active threads store to shared memory.
    if (vo >= midpoint && vo < active_groups && is_valid_vi) {
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      convert_from_float(
          *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]),
          out);
#else
      *reinterpret_cast<V_vec *>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
    }
    __syncthreads();

    // The bottom warps update their values.
    if (vo < midpoint && is_valid_vi) {
      out = add(*reinterpret_cast<V_vec const *>(&out_smem[vo * Dh + vi]), out);
    }
    __syncthreads();
  }

  //// part_out（_sum/_max） 存入全局内存中 (2651-2697)
  // partial_out [seq_len_tile, bsz, num_head, dim_head]
  // partial_max/partial_sum [bsz, num_head, seq_len_tile]
  auto const bhi_seq_len_tile = bhi * params.seq_len_tile;
  if (vo == 0 && is_valid_vi) {
    const int bhvi = bhi * Dh + vi;
    if (!DO_MULTI_BLOCK){
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
      V_vec final_out;
      convert_from_float(final_out, out);
      store_func.template store<V_vec>(final_out, bhvi);
#else
      store_func.template store<V_vec>(out, bhvi);
#endif
    }
    else{
      // for write partial output to partial_out
      int partial_out_offset = c_tile * params.batch_size * num_heads * Dh;
      // for write partial statistics to partial_max and partial_sum
      int partial_stats_offset = bhi_seq_len_tile + c_tile;

      // This makes sure we have coalesced memory access.
      V_vec partial_out;
      convert_from_float(partial_out, out);
      *reinterpret_cast<V_vec *>(&params.partial_out[partial_out_offset + bhvi]) =
          partial_out;

      *reinterpret_cast<float *>(&params.partial_max[partial_stats_offset]) =
          qk_max;

      *reinterpret_cast<float *>(&params.partial_sum[partial_stats_offset]) = sum;
    }
  }

  //// reduce part_out to out (2699-2863)
  if (DO_MULTI_BLOCK) {
    // cuda::thread_scope_device	与发起线程同一GPU设备中的所有CUDA线程同步。
    cuda::atomic_ref<int, cuda::thread_scope_device> count_ref{
        params.block_counter[bhi]};
    bool is_last_block_runtime{false};
    if (tid == 0) {
      if (count_ref.fetch_add(1, cuda::memory_order_acq_rel) ==
          (gridDim.z - 1)) {
        is_last_block_runtime = true;
      }
    }

    if (__syncthreads_or(is_last_block_runtime)) {
      float final_max = -FLT_MAX;
      float thread_partial_max = -FLT_MAX;
      thread_partial_max =
          params.partial_max[bhi_seq_len_tile + min(tid, gridDim.x - 1)];

      __syncthreads();

      typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      final_max = BlockReduce(temp_storage)
                      .Reduce(thread_partial_max, cub::Max(), gridDim.x);

      __shared__ float final_max_smem;
      if (tid == 0) {
        final_max_smem = final_max;
      }
      __syncthreads();
      // Finish the final_max computation
      final_max = final_max_smem;

      float final_sum = 0.f;
      if (tid < gridDim.x) {
        thread_partial_max = params.partial_max[bhi_seq_len_tile + tid];
        auto const thread_partial_sum =
            params.partial_sum[bhi_seq_len_tile + tid];
        final_sum +=
            __expf(thread_partial_max - final_max) * thread_partial_sum;
      }
      // Finish the final_sum computation
      final_sum =
          block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], final_sum);

      // partial_out [seq_len_tile, bsz, num_head, dim_head]
      T *out_oi_smem = reinterpret_cast<T *>(smem_);
      auto const o_idx = chunk_index<T, V_vec, THREADS_PER_VALUE>(tid);
      auto const oo = o_idx.x;
      auto const oi = o_idx.y;
      
      V_vec thread_accumulated_out;
      zero(thread_accumulated_out);
      
      for (int tile_idx = oo; tile_idx < gridDim.x; tile_idx += V_PER_ITER) {
        // Load partial output
        int thread_partial_out_offset =
            tile_idx * params.batch_size * num_heads * Dh;
        // Load partial max (different to thread_partial_max since the threadIdx
        // rule changes here)
        float thread_partial_max_for_out =
            params.partial_max[bhi_seq_len_tile + tile_idx];
        // Load the partial outputs.
        V_vec thread_partial_out = *reinterpret_cast<V_vec const *>(
            &params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);
        // Apply the correction factor.
        float factor_compute = __expf(thread_partial_max_for_out - final_max);
        thread_partial_out =
            mul<V_vec, float, V_vec>(factor_compute, thread_partial_out);
        thread_accumulated_out =
            add(thread_partial_out, thread_accumulated_out);
      }
#pragma unroll
      for (int active_groups = V_PER_ITER; active_groups >= 2;
           active_groups /= 2) {
        // The midpoint in the number of active groups.
        int midpoint = active_groups / 2;

        // The upper part of active threads store to shared memory.
        if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh)) {
          *reinterpret_cast<V_vec *>(&out_oi_smem[(oo - midpoint) * Dh + oi]) =
              thread_accumulated_out;
        }
        __syncthreads();

        // The bottom warps update their values.
        if (oo < midpoint && (Dh == Dh_MAX || oi < Dh)) {
          thread_accumulated_out =
              add(thread_accumulated_out,
                  *reinterpret_cast<V_vec const *>(&out_oi_smem[oo * Dh + oi]));
        }
        __syncthreads();
      }

      if (oo == 0 && (Dh == Dh_MAX || oi < Dh)) {
        float const inv_sum = __fdividef(1.f, final_sum + 1.e-6f);
        thread_accumulated_out =
            mul<V_vec, float, V_vec>(inv_sum, thread_accumulated_out);
        // *reinterpret_cast<V_vec *>(static_cast<T *>(params.out) +
        //                            (bhi * Dh + oi)) = thread_accumulated_out;
        store_func.template store<V_vec>(thread_accumulated_out, bhi * Dh + oi);
      }

      if (tid == 0) {
        params.block_counter[bhi] = 0;
      }
    }
  }
}

template <typename T, unsigned Dh, bool DO_MULTI_BLOCK>
inline size_t smem_size_in_bytes(
    Masked_multihead_attention_params<T> const &params, int threads_per_block) {
  // The amount of shared memory needed to store the Q*K^T values in float.
  unsigned const max_timesteps =
      DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep;
  // 这里想表达elts是16B为单位的，就是4个float。
  std::size_t const qk_elts = static_cast<std::size_t>(
      div_up(max_timesteps + 1, 4u));  // explicit cast because of the sign
  std::size_t const qk_sz = qk_elts * 16;

  // The extra memory needed if we are not using floats for the final logits.
  // 我觉得固定用float存logits就挺好，不过可能多点也没事。一个SM起一个Block的情况下。
  size_t logits_sz = 0;
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
  if (sizeof(T) != 4) {
    logits_sz = qk_elts * 4 * sizeof(T);
  }
#endif

  // The total size needed during softmax.
  size_t softmax_sz = qk_sz + logits_sz;

  unsigned constexpr threads_per_value = getThreadsPerValue<T, Dh>();

  // The number of partial rows to reduce in the final reduction.
  int rows_per_red = threads_per_block / threads_per_value;
  // The amount of storage needed to finalize the outputs.
  size_t red_sz = rows_per_red * Dh * sizeof(T) / 2;

  size_t out_oi_sz = 0;
  if (params.multi_block_mode) {
    // The size for partial output reduction computation.
    out_oi_sz = params.seq_len_tile * Dh * sizeof(T);
  }

  // The max.
  return max(max(softmax_sz, red_sz), out_oi_sz);
}

template <typename T, unsigned Dh>
inline void get_seq_len_tile(Masked_multihead_attention_params<T> const &params,
                             unsigned blocks_per_sm,
                             unsigned block_size) {
  unsigned balanced_seq_len_tile =
      div_up(getMultiProcessorCount() * blocks_per_sm,
             params.batch_size * params.num_heads);

  unsigned const threads_per_value = getThreadsPerValue<T, Dh>();
  /// Make sure that each block at least processes one loop of kv (unroll size
  /// is default at 8)
  unsigned const seq_len_per_kv_loop =
      div_up(block_size, threads_per_value) * 8;
  unsigned max_seq_len_tile =
      min(params.max_seq_len_tile,
          div_up(params.timestep + 1, seq_len_per_kv_loop));

  // seq_len_tile的下界是在seq超长的情况下，至少需要这些block，可能一个wave解决不了。
  // seq_len_tile的上界是在seq超短的情况下，一个wave能启动的seq维block数，如果硬件还是太充足，优先保证一个block的工作长度。
  params.seq_len_tile = std::clamp(
      balanced_seq_len_tile, params.min_seq_len_tile, max_seq_len_tile);

  params.timesteps_per_block = div_up(params.timestep + 1, params.seq_len_tile);

  params.multi_block_mode = (params.seq_len_tile > 1);
}

#define MMHA_LAUNCH_KERNEL(DYNAMIC_THDS_PER_BLOCK)                     \
  case DYNAMIC_THDS_PER_BLOCK:                                         \
    smem_sz = smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>(               \
        params, DYNAMIC_THDS_PER_BLOCK);                               \
    if (smem_sz >= 46 * 1024) {                                        \
      cudaFuncSetAttribute(                                            \
          masked_multihead_attention_kernel<T,                         \
                                            Dh,                        \
                                            DYNAMIC_THDS_PER_BLOCK,    \
                                            decltype(load_func),       \
                                            decltype(store_func),      \
                                            DO_MULTI_BLOCK>,           \
          cudaFuncAttributeMaxDynamicSharedMemorySize,                 \
          smem_sz);                                                    \
    }                                                                  \
    masked_multihead_attention_kernel<T,                               \
                                      Dh,                              \
                                      DYNAMIC_THDS_PER_BLOCK,          \
                                      decltype(load_func),             \
                                      decltype(store_func),            \
                                      DO_MULTI_BLOCK>                  \
        <<<grid, DYNAMIC_THDS_PER_BLOCK, smem_sz, dev_ctx.stream()>>>( \
            params, load_func, store_func);                            \
    break;

#define MMHA_LAUNCH_CHECK(DYNAMIC_THDS_PER_BLOCK)                              \
  std::size_t const dynamic_smem_sz{smem_size_in_bytes<T, Dh, DO_MULTI_BLOCK>( \
      params, DYNAMIC_THDS_PER_BLOCK)};                                        \
  if (dynamic_smem_sz >= 46 * 1024) {                                          \
    cudaError_t res = cudaFuncSetAttribute(                                    \
        masked_multihead_attention_kernel<T,                                   \
                                          Dh,                                  \
                                          DYNAMIC_THDS_PER_BLOCK,              \
                                          decltype(load_func),                 \
                                          decltype(store_func),                \
                                          DO_MULTI_BLOCK>,                     \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                           \
        dynamic_smem_sz);                                                      \
    PADDLE_ENFORCE_EQ(                                                         \
        res,                                                                   \
        cudaSuccess,                                                           \
        common::errors::PreconditionNotMet(                                    \
            "Sequence Length is too long for the MMHA kernel."));              \
  }                                                                            \
  check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(              \
      &available_blocks,                                                       \
      masked_multihead_attention_kernel<T,                                     \
                                        Dh,                                    \
                                        DYNAMIC_THDS_PER_BLOCK,                \
                                        decltype(load_func),                   \
                                        decltype(store_func),                  \
                                        DO_MULTI_BLOCK>,                       \                                             
      DYNAMIC_THDS_PER_BLOCK,                                                  \
      dynamic_smem_sz));

// if resources are not enough to launch 512 threads per block, we will fallback
// to 256.
#define MMHA_512_BLOCKSIZE_CHECK() \
  MMHA_LAUNCH_CHECK(512);          \
  if (available_blocks <= 0) {     \
    MMHA_LAUNCH_CHECK(256);        \
    dynamic_block_size = 256;      \
  } else {                         \
    dynamic_block_size = 512;      \
  }

// if resources are not enough to launch 1024 threads per block, we will
// fallback to 512.
#define MMHA_1024_BLOCKSIZE_CHECK() \
  MMHA_LAUNCH_CHECK(1024);          \
  if (available_blocks > 0) {       \
    dynamic_block_size = 1024;      \
  } else {                          \
    MMHA_512_BLOCKSIZE_CHECK();     \
  }

/// 在硬件限制下，我们设置一个尽可能占用更多资源的block来处理一个seq_len_tile。
/// 同样在硬件限制下，我们让seq_len_tile尽可能多。
/// 一个wave结束战斗。
/// 按max_seq_len_tile的定义，seq不太长的情况下，一个SM只会启动一个block。
template <typename T,
          unsigned Dh,
          unsigned THDS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc,
          bool DO_MULTI_BLOCK>
void getGrid(Masked_multihead_attention_params<T> &params,
             LoadFunc load_func,
             StoreFunc store_func) {
  params.seq_len_tile = 1;
  int const kernel_total_blocks = params.num_heads * params.batch_size;
  // Don't tune the block size if batch*head is large enough
  // THDS_PER_BLOCK(256) * 4 = 1024
  if (!DO_MULTI_BLOCK && kernel_total_blocks >= getMultiProcessorCount() * 4) {
    return;
  }

  int num_blocks_per_sm = -1;
  // Set 0 dynamic shared memory size as we need the number of available blocks
  // limited by registers.
  check_cuda_error(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm,
      masked_multihead_attention_kernel<T,
                                        Dh,
                                        THDS_PER_BLOCK,
                                        decltype(load_func),
                                        decltype(store_func),
                                        DO_MULTI_BLOCK>,
      THDS_PER_BLOCK,
      0));

  int block_size_factor = min(
      div_up(getMultiProcessorCount() * num_blocks_per_sm, kernel_total_blocks),
      num_blocks_per_sm);

  int dynamic_block_size = min(1024, THDS_PER_BLOCK * block_size_factor);

  int available_blocks = -1;
  if (dynamic_block_size < 512) {
    MMHA_LAUNCH_CHECK(256);
    dynamic_block_size = 256;
  } else if (dynamic_block_size < 1024) {
    MMHA_512_BLOCKSIZE_CHECK();
  } else if (dynamic_block_size == 1024) {
    MMHA_1024_BLOCKSIZE_CHECK();
  }

  get_seq_len_tile<T, Dh>(params, available_blocks, dynamic_block_size);

  params.dynamic_block_size = dynamic_block_size;
}

template <typename T,
          unsigned Dh,
          unsigned THREADS_PER_BLOCK,
          typename LoadFunc,
          typename StoreFunc,
          bool DO_MULTI_BLOCK>
void DispatchWithDh(const phi::GPUContext &dev_ctx,
                    Masked_multihead_attention_params<T> &params,
                    LoadFunc load_func,
                    StoreFunc store_func) {
  getGrid<T,
          Dh,
          THREADS_PER_BLOCK,
          decltype(load_func),
          decltype(store_func),
          DO_MULTI_BLOCK>(params, load_func, store_func);
  dim3 grid{params.seq_len_tile, params.num_heads, params.batch_size};
  size_t smem_sz = 0;
  switch (params.dynamic_block_size) {
    MMHA_LAUNCH_KERNEL(256)
    MMHA_LAUNCH_KERNEL(512)
    MMHA_LAUNCH_KERNEL(1024)
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Block Size = %d is unsupport!", params.dynamic_block_size));
  }
}

template <typename T,
          typename LoadFunc,
          typename StoreFunc,
          bool DO_MULTI_BLOCK>
void DispatchWithLSFunc(const phi::GPUContext &dev_ctx,
                        Masked_multihead_attention_params<T> &params,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  switch (params.dim_head) {
    case 32:
      DispatchWithDh<T,
                     32,
                     256,
                     decltype(load_func),
                     decltype(store_func),
                     DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
      break;
    case 64:
      DispatchWithDh<T,
                     64,
                     256,
                     decltype(load_func),
                     decltype(store_func),
                     DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
      break;
    case 128:
      DispatchWithDh<T,
                     128,
                     256,
                     decltype(load_func),
                     decltype(store_func),
                     DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
      break;
    case 256:
      DispatchWithDh<T,
                     256,
                     256,
                     decltype(load_func),
                     decltype(store_func),
                     DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
      break;
    // case 48:
    //   DispatchWithDh<T,
    //                  48,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 80:
    //   DispatchWithDh<T,
    //                  80,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 96:
    //   DispatchWithDh<T,
    //                  96,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 104:
    //   DispatchWithDh<T,
    //                  104,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 112:
    //   DispatchWithDh<T,
    //                  112,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 144:
    //   DispatchWithDh<T,
    //                  144,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 160:
    //   DispatchWithDh<T,
    //                  160,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 192:
    //   DispatchWithDh<T,
    //                  192,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    // case 224:
    //   DispatchWithDh<T,
    //                  224,
    //                  256,
    //                  decltype(load_func),
    //                  decltype(store_func),
    //                  DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
    //   break;
    default:
      PADDLE_THROW(common::errors::Unimplemented("Dim_head = %d is unsupport!",
                                                 params.dim_head));
  }
}

template <typename T, bool DO_MULTI_BLOCK = false>
void DispatchWithMultiBlock(
    const phi::GPUContext &dev_ctx,
    const phi::DenseTensor &qkv_tensor,
    Masked_multihead_attention_params<T> &params,
    phi::DenseTensor *out_tensor,
    const phi::DenseTensor *dequant_qkv_scales = nullptr,
    const float quant_fmha_out_scale = -1,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0f,
    const float quant_min_bound = -127.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * params.num_heads * params.dim_head);
    MMHAStore<T, int8_t> store_func(out_tensor->data<int8_t>(),
                                    quant_round_type,
                                    quant_fmha_out_scale,
                                    quant_max_bound,
                                    quant_min_bound);
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, int8_t> store_func(out_tensor->data<int8_t>(),
                                    quant_round_type,
                                    quant_fmha_out_scale,
                                    quant_max_bound,
                                    quant_min_bound);
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * params.num_heads * params.dim_head);
    MMHAStore<T> store_func(out_tensor->data<T>());
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T> store_func(out_tensor->data<T>());
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  }
}

template <typename T, bool DO_MULTI_BLOCK = false>
void DispatchWithMultiBlock(
    const phi::GPUContext &dev_ctx,
    const phi::DenseTensor &qkv_tensor,
    const phi::DenseTensor &shift,
    const phi::DenseTensor &smooth,
    Masked_multihead_attention_params<T> &params,
    phi::DenseTensor *out_tensor,
    const phi::DenseTensor *dequant_qkv_scales = nullptr,
    const float quant_fmha_out_scale = -1,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0f,
    const float quant_min_bound = -127.0f) {
  if (dequant_qkv_scales != nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * params.num_heads * params.dim_head);
    MMHAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          params.num_heads * params.dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  } else if (dequant_qkv_scales == nullptr && quant_fmha_out_scale > 0) {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, int8_t, true> store_func(out_tensor->data<int8_t>(),
                                          shift.data<T>(),
                                          smooth.data<T>(),
                                          params.num_heads * params.dim_head,
                                          quant_round_type,
                                          quant_fmha_out_scale,
                                          quant_max_bound,
                                          quant_min_bound);
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  } else if (dequant_qkv_scales != nullptr && quant_fmha_out_scale <= 0) {
    MMHALoad<T, int32_t> load_func(qkv_tensor.data<int32_t>(),
                                   dequant_qkv_scales->data<float>(),
                                   3 * params.num_heads * params.dim_head);
    MMHAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     params.num_heads * params.dim_head);
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
  } else {
    MMHALoad<T> load_func(qkv_tensor.data<T>());
    MMHAStore<T, T, true> store_func(out_tensor->data<T>(),
                                     shift.data<T>(),
                                     smooth.data<T>(),
                                     params.num_heads * params.dim_head);
    DispatchWithLSFunc<T,
                       decltype(load_func),
                       decltype(store_func),
                       DO_MULTI_BLOCK>(dev_ctx, params, load_func, store_func);
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

// 大致意思就是：从硬件角度，一个wave处理完全部任务的情况下，最大的seq_len_tile
// 实际上最后也是一个wave处理完的，只要一个wave。
// 假定了一个SM只能有一个block
/// TODO:
/// 整个attention实现的思路就是要block尽量大(占用资源尽可能多)，所以说这个假定什么时候不合理呢？
// 就是一个block有1024线程，但是寄存器和共享内存都只占了不到一半。这种情况下，A100说不定能启动
// 两个block/SM 但是很难说这种情况会性能好，所以暂时来讲，这个假定还算合理。
int getMaxNumSeqLenTile(int batch_size, int num_heads) {
  return div_up(getMultiProcessorCount(), batch_size * num_heads);
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
                       NormalVersion) {
  // input_tensor [batch_size, (num_heads + 2 * kv_num_heads) * head_dim].
  const auto &x_dims = x.dims();
  int batch_size = x_dims[0];  // batch * beam_width
  // cache_kv [2, batch_size, num_heads, max_seq_len, head_dim].
  int cache_bsz = cache_kv.dims()[1];
  int kv_num_heads = cache_kv.dims()[2];
  int max_seq_len = cache_kv.dims()[3];
  int dim_head = cache_kv.dims()[4];

  int timestep = max_seq_len;
  float inv_sqrt_dh = 1. / sqrt(dim_head);

  int num_heads = x.dims()[x.dims().size() - 1] / dim_head - kv_num_heads * 2;

  Masked_multihead_attention_params<T> params;
  bool mask_broadcast_num_heads = true;

  params.add_qkv_bias = false;
  if (bias) {
    params.add_qkv_bias = true;
    params.qkv_bias = const_cast<T *>(bias->data<T>());
  }

  if (src_mask) {
    if (src_mask->dims()[1] == 1) {
      mask_broadcast_num_heads = true;
    } else if (src_mask->dims()[1] == num_heads) {
      mask_broadcast_num_heads = false;
    } else {
      PADDLE_THROW(errors::InvalidArgument(
          "Unknow dimension for attn_mask, the num_heads(2nd) "
          "dimension is invalid, it should be 1 or num_heads(%d), "
          "but got %d",
          num_heads,
          src_mask->dims()[1]));
    }
    params.attn_mask = src_mask->data<T>();
    // 目前paddlenlp用这个算子的时候
    // 给的src_mask也是max_seq_len，不是实际cache长度。
    timestep = src_mask->dims()[3] - 1;
  }

  if (out_scale > 0) {
    dev_ctx.template Alloc<int8_t>(out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }
  if (sequence_lengths) {
    params.sequence_lengths = sequence_lengths->data<int>();
  }

  if (cum_offsets) {
    PADDLE_THROW(common::errors::PermissionDenied(
        "Current mmha kernel does not support cum_offsets param."));
  }

  if (rotary_emb_dims > 0) {
    params.rotary_emb = rotary_tensor->data<float>();
  } else {
    params.rotary_emb = nullptr;
  }

  if (beam_cache_offset) {
    params.beam_cache_offset = beam_cache_offset->data<int>();
    params.beam_width = beam_cache_offset->dims()[1];
  }

  params.mask_broadcast_num_heads = mask_broadcast_num_heads;
  params.cache_kv = const_cast<T *>(cache_kv_out->data<T>());
  params.neox_rotary_style = use_neox_rotary_style;
  params.batch_size = batch_size;
  params.cache_batch_size = cache_bsz;
  params.num_heads = num_heads;
  params.num_heads_kv = kv_num_heads;
  params.timestep = timestep;
  params.max_seq_length = max_seq_len;
  params.inv_sqrt_dh = inv_sqrt_dh;
  params.rotary_emb_dims = rotary_emb_dims;
  params.dim_head = dim_head;

  /// TODO:此类函数paddle应该有实现，可以换成paddle的版本。
  int mMaxSharedMemoryPerBlockOptin = getMaxSharedMemoryPerBlockOptin();
  // 从硬件角度，不管怎样安排seq_len_tile,
  // 一个block最大只能处理这些数据，所以至少需要这么多block
  int estimated_min_multi_block_count = estimate_min_multi_block_count<T>(
      timestep, mMaxSharedMemoryPerBlockOptin - 2048);
  // 这里params.batch_size已经是 real_bsz *
  // beam_width了。且我们对beam并不感兴趣。
  const int max_num_seq_len_tiles =
      std::max(getMaxNumSeqLenTile(batch_size, num_heads),
               estimated_min_multi_block_count);
  const int min_num_seq_len_tiles = estimated_min_multi_block_count;
  params.max_seq_len_tile = max_num_seq_len_tiles;
  params.min_seq_len_tile = min_num_seq_len_tiles;

  // 这样初始化会造成什么影响？
  params.timesteps_per_block = div_up(timestep + 1, min_num_seq_len_tiles);

  bool const multi_block_mode = max_num_seq_len_tiles > 1;
  params.multi_block_mode = multi_block_mode;

  if (multi_block_mode) {
    // partial_max/partial_sum [bsz, num_head, max_seq_len_tile]
    phi::DenseTensor partial_sum;
    partial_sum.Resize({{batch_size, num_heads, max_num_seq_len_tiles}});
    dev_ctx.template Alloc<float>(&partial_sum,
                                  partial_sum.numel() * sizeof(float));
    params.partial_sum = partial_sum.data<float>();

    phi::DenseTensor partial_max;
    partial_max.Resize({{batch_size, num_heads, max_num_seq_len_tiles}});
    dev_ctx.template Alloc<float>(&partial_max,
                                  partial_max.numel() * sizeof(float));
    params.partial_max = partial_max.data<float>();

    // partial_out [max_seq_len_tile, bsz, num_head, dim_head]
    phi::DenseTensor partial_out;
    partial_out.Resize(
        {{max_num_seq_len_tiles, batch_size, num_heads, dim_head}});
    dev_ctx.template Alloc<float>(&partial_out,
                                  partial_out.numel() * sizeof(float));
    params.partial_out = partial_out.data<float>();

    // block_counter [batch_size, num_heads]
    phi::DenseTensor block_counter;
    block_counter.Resize({{batch_size, num_heads}});
    dev_ctx.template Alloc<int>(&block_counter,
                                block_counter.numel() * sizeof(int));
    params.block_counter = block_counter.data<int>();

    if (out_shift) {
      DispatchWithMultiBlock<T, true>(dev_ctx,
                                      x,
                                      *(out_shift.get_ptr()),
                                      *(out_smooth.get_ptr()),
                                      params,
                                      out,
                                      qkv_out_scale.get_ptr(),
                                      out_scale,
                                      quant_round_type,
                                      quant_max_bound,
                                      quant_min_bound);
    } else {
      DispatchWithMultiBlock<T, true>(dev_ctx,
                                      x,
                                      params,
                                      out,
                                      qkv_out_scale.get_ptr(),
                                      out_scale,
                                      quant_round_type,
                                      quant_max_bound,
                                      quant_min_bound);
    }
  } else {
    if (out_shift) {
      DispatchWithMultiBlock<T, false>(dev_ctx,
                                       x,
                                       *(out_shift.get_ptr()),
                                       *(out_smooth.get_ptr()),
                                       params,
                                       out,
                                       qkv_out_scale.get_ptr(),
                                       out_scale,
                                       quant_round_type,
                                       quant_max_bound,
                                       quant_min_bound);
    } else {
      DispatchWithMultiBlock<T, false>(dev_ctx,
                                       x,
                                       params,
                                       out,
                                       qkv_out_scale.get_ptr(),
                                       out_scale,
                                       quant_round_type,
                                       quant_max_bound,
                                       quant_min_bound);
    }
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
        PADDLE_THROW(common::errors::InvalidArgument(
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
