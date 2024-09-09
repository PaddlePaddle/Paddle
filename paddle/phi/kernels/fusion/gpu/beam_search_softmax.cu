// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/kernels/fusion/cutlass/utils/cuda_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"


namespace phi {
namespace fusion {

#define FLT_MAX 1e38
// #define DEBUG_BEAM_SEARCH_SOFTMAX

static constexpr int kBlockSizeForSmallBeamWidth = 256;
static constexpr int kMaxVocabPartForStage1FastKernel = 128;

#define CASE_K(K)                                        \
  case K:                                                \
    invokeTopKSoftMaxLauncher<T, Context, 2 * K, GROUP>( \
        dev_ctx, params, beam_group_idx, stream);        \
    break

#define DISPATCH_COMPUTE_PARTS_K(K)                      \
   case K:                                               \
    ComputeVocParts<T, 2 * K>(params);                   \
    break

template <typename T>
struct BeamSearchParams {
  // Scalar values
  int batch_size{0};
  int beam_width{0};
  int beam_group_size{0};
  int beam_group_idx{0};

  int vocab_size{0};
  int dec_stride{0};
  int max_seq_len{0};
  int end_ids_len{0};

  bool fuse_softmax{true};
  bool early_stop{false};

  int voc_parts{0};
  bool use_fast_kernel{true};
  int max_smem_per_block{0};

  T *logits{nullptr};
  const int *step_ids{nullptr};               // [BS * BM, 1]
  const int *seq_lens{nullptr};               // [BS * BM, 1]

  const int *max_dec_lens{nullptr};
  const int *end_ids{nullptr};

  const T *cum_scores{nullptr};
  const int *block_tables{nullptr};
  const int *beam_cache_ids{nullptr};         

  const float *length_penalty{nullptr};       // [BS, 1]
  const float *diversity_penalty{nullptr};    // [BS, 1]

  bool *stop_flags{nullptr};                  // [BS, 1]
  int *cache_ids_out{nullptr};                // [BS * BM, max_dec_len]
  bool *beam_finished{nullptr};               // [BS * BM, 1]
  int *block_tables_out{nullptr};             // [BS * BM, max_seq_len]
  T *cum_scores_out{nullptr};                 // [BS * BM, 1]
  int *beam_hyps_out{nullptr};                // [BS * BM, max_dec_len]
  T *beam_hyps_score_out{nullptr};            // [BS * BM, 1]

  // func out
  int *next_tokens{nullptr};
  int *parent_ids{nullptr};

  // workspace
  int *tmp_ids{nullptr};
  T *tmp_vals{nullptr};
  T *tmp_buffer{nullptr};
};

template <typename T,
          typename U,
          typename = std::enable_if_t<std::is_integral<T>::value>,
          typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

__device__ bool is_in_end(const int id, const int *end_ids, int length) {
  bool flag = false;
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return flag;
}

template <typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob,
                                                  int length,
                                                  float length_penalty) {
  // score = log(prob) / (length)^length_penalty.
  if (length_penalty == 0.0f || length == 1) {
    return log_prob;
  }
  return log_prob / static_cast<T>(powf(length, length_penalty));
}

// <<<batch_size, beam_group_size>>>
template <typename T, int K>
__global__ void apply_group_diversity_penalty(BeamSearchParams<T> params,
                                              const int batch_size,
                                              const int beam_width,
                                              const int beam_group_idx,
                                              const int vocab_size) {
  const int beam_group_size = K / 2;
  const int batch_idx = blockIdx.x;
  const int beam_group_sub_idx = threadIdx.x;
  const bool *beam_finished = params.beam_finished + batch_idx * beam_width;
  T *logtis = params.logits + batch_idx * beam_width * vocab_size +
              beam_group_idx * beam_group_size * vocab_size +
              beam_group_sub_idx * vocab_size;
  int *next_tokens = params.next_tokens + batch_idx * beam_width;
  // apply previous group token ids penalty
#pragma unroll
  for (int token_idx = 0; token_idx < beam_group_idx * beam_group_size;
       ++token_idx) {
    const bool finished = beam_finished[token_idx];
    if (!finished) {
      const int token_id = next_tokens[token_idx];
      logtis[token_id] -= params.diversity_penalty[batch_idx];
    }
  }
}

struct DySoftMaxStruct {
  float logit;
  float score;
};

__device__ __forceinline__ DySoftMaxStruct
reduce_softmax_op(DySoftMaxStruct a, DySoftMaxStruct b) {
  bool a_bigger = (a.logit > b.logit);
  DySoftMaxStruct bigger_m = a_bigger ? a : b;
  DySoftMaxStruct smaller_m = a_bigger ? b : a;
  DySoftMaxStruct res;
  res.score =
      bigger_m.score + smaller_m.score * expf(smaller_m.logit - bigger_m.logit);
  res.logit = bigger_m.logit;
  return res;
}

template <typename T>
struct BeamHypothesis {
  T score;
  int *seq;
  int seq_len;

  __device__ __forceinline__ void init(int *_seq,
                                       T _score,
                                       const int _max_seq_len) {
    seq = _seq;
    score = _score;
    seq_len = _max_seq_len;
  }
};

template <typename T, int K>
struct BeamHypothesesTopK {
  BeamHypothesis<T> hyps[K];
  int max_dec_len;

  __device__ __forceinline__ void init(int *_beam_hyps,
                                       T *_beam_hyps_score,
                                       const int _max_dec_len) {
    max_dec_len = _max_dec_len;
    for (int i = 0; i < K; i++) {
      // 使用默认构造函数创建默认的 BeamHypothesis 对象
      hyps[i].init(
          _beam_hyps + i * _max_dec_len, _beam_hyps_score[i], _max_dec_len);
    }
  }

  __device__ void insert(const int *token_ids,
                         int step,
                         int cur_token_id,
                         T score) {
    if (score > get_worst_score()) {
      for (int i = 0; i < step; i++) {
        hyps[K - 1].seq[i] = token_ids[i];
      }
      hyps[K - 1].seq[step] = cur_token_id;
      hyps[K - 1].score = score;

      for (int k = K - 2; k >= 0; --k) {
        if (hyps[k + 1].score > hyps[k].score) {
          T tmp_score = hyps[k].score;
          hyps[k].score = hyps[k + 1].score;
          hyps[k + 1].score = tmp_score;

          int tmp_val;
          for (int i = 0;
               i <= step && (hyps[k + 1].seq[i] > 0 || hyps[k].seq[i] > 0);
               i++) {
            tmp_val = hyps[k + 1].seq[i];
            hyps[k + 1].seq[i] = hyps[k].seq[i];
            hyps[k].seq[i] = tmp_val;
          }
        }
      }
    }
  }

  __device__ __forceinline__ T get_worst_score() { return hyps[K - 1].score; }
};

template <typename T, int K>
struct TopK {
  int ids[K];
  T vals[K];
  int parent_ids[K];

  __device__ __forceinline__ void insert(T elem, int elem_id) {
    if (elem > vals[K - 1] || (ids[K - 1] == -1) ||
        ((elem == vals[K - 1]) && (elem_id < ids[K - 1]))) {
      vals[K - 1] = elem;
      ids[K - 1] = elem_id;
    }

    for (int k = K - 2; k >= 0; --k) {
      if ((vals[k + 1] > vals[k]) || (ids[k] == -1) ||
          ((vals[k + 1] == vals[k]) && (ids[k + 1] < ids[k]))) {
        T tmp_val = vals[k];
        int tmp_id = ids[k];
        vals[k] = vals[k + 1];
        ids[k] = ids[k + 1];
        vals[k + 1] = tmp_val;
        ids[k + 1] = tmp_id;
      }
    }
  }

  __device__ __forceinline__ void insert(T elem, int elem_id, int parent_id) {
    if (elem > vals[K - 1] || (ids[K - 1] == -1) ||
        ((elem == vals[K - 1]) && (elem_id < ids[K - 1]))) {
      vals[K - 1] = elem;
      ids[K - 1] = elem_id;
      parent_ids[K - 1] = parent_id;
    }

    for (int k = K - 2; k >= 0; --k) {
      if ((vals[k + 1] > vals[k]) || (ids[k] == -1) ||
          ((vals[k + 1] == vals[k]) && (ids[k + 1] < ids[k]))) {
        T tmp_val = vals[k];
        int tmp_id = ids[k];
        int parent_id2 = parent_ids[k];
        vals[k] = vals[k + 1];
        ids[k] = ids[k + 1];
        parent_ids[k] = parent_ids[k + 1];
        vals[k + 1] = tmp_val;
        ids[k + 1] = tmp_id;
        parent_ids[k + 1] = parent_id2;
      }
    }
  }
};

template <typename T, int K>
__device__ __forceinline__ TopK<T, K> reduce_topk_op(const TopK<T, K> &a,
                                                     const TopK<T, K> &b) {
  TopK<T, K> res = a;
  for (int i = 0; i < K; ++i) res.insert(b.vals[i], b.ids[i]);
  return res;
}

template <typename T, int K>
struct TopKSoftMax {
  DySoftMaxStruct softmax_md;
  TopK<T, K> topk;
};

template <typename T, int K>
__device__ __forceinline__ TopKSoftMax<T, K> reduce_topk_softmax_op(
    const TopKSoftMax<T, K> &a, const TopKSoftMax<T, K> &b) {
  TopKSoftMax<T, K> res;
  // max_logit in block
  res.softmax_md = reduce_softmax_op(a.softmax_md, b.softmax_md);
  res.topk = reduce_topk_op(a.topk, b.topk);
  return res;
}

struct __align__(8) MD {
  float m;
  float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b) {
  bool const isABigger = a.m > b.m;
  MD const bigger = isABigger ? a : b;
  MD const smaller = isABigger ? b : a;
  MD res{bigger.m, bigger.d + smaller.d * __expf(smaller.m - bigger.m)};
  return res;
}

template <typename T, int K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE, 1) __global__
    void beam_search_softmax_topk_stage1_fast(const T *logits,
                                              float *tmp_buffer,
                                              const int *end_ids,
                                              const bool *beam_finished,
                                              const int *seq_lens,
                                              int beam_width,
                                              int beam_group_idx,
                                              int vocab_size,
                                              int vocab_chunk_size) {
  constexpr int PACKED_TOP_KMD_SIZE = 2 * K + 2;
  const int beam_group_size = K / 2;
  const int tid = threadIdx.x;
  const int group_beam_batch_id = blockIdx.x;
  const int batch_id = group_beam_batch_id / beam_group_size;
  const int beam_group_sub_id = group_beam_batch_id % beam_group_size;
  const int beam_batch_id = batch_id * beam_width +
                            beam_group_idx * beam_group_size +
                            beam_group_sub_id;

  const int seq_len = seq_lens[beam_batch_id];
  const bool finished = beam_finished[beam_batch_id];

  if (seq_len < 0 || finished) {
    return;
  }

  const int section_start = vocab_chunk_size * blockIdx.y;
  const int section_end =
      std::min(section_start + vocab_chunk_size, vocab_size);
  const int valid_smem_length = section_end - section_start;
  T const MAX_T_VAL = 1e38;

  // Load element from logits to smemLogProbs, doing reduce_md and argmax
  // meanwhile Each thread is responsible for `vocab_chunk_size /
  // THREADBLOCK_SIZE` elements
  extern __shared__ char smem[];
  T *smemLogProbs = reinterpret_cast<T *>(smem);

  MD partial_md{-MAX_T_VAL, 0.0f};

  using KVPair = cub::KeyValuePair<int, T>;
  KVPair topKVPairPartial{vocab_size - 1, -MAX_T_VAL};
  cub::ArgMax argmax;

  T const *local_logits = logits + beam_batch_id * vocab_size;
#pragma unroll 1
  for (int i = section_start + tid; i < section_end; i += THREADBLOCK_SIZE) {
    T const val = local_logits[i];
    const int smem_index = i - section_start;
    smemLogProbs[smem_index] = val;
    MD new_elem_md{val, 1.0F};
    partial_md = reduce_md_op(partial_md, new_elem_md);
    KVPair new_elem_topk{smem_index, val};
    topKVPairPartial = argmax(topKVPairPartial, new_elem_topk);
  }
  __syncthreads();

  // Search the top 2K elements among `vocab_chunk_size` elements of this
  // ThreadBlock and write into smemOutput
  __shared__ float smemOutput[PACKED_TOP_KMD_SIZE];
  __shared__ int threadToUpdate;

  using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;
  using BlockReduceTopK = cub::BlockReduce<KVPair, THREADBLOCK_SIZE>;

  __shared__ union {
    typename BlockReduceTopK::TempStorage topk;
    typename BlockReduceMD::TempStorage md;
  } smemReduceBuffer;

  for (int i = 0; i < 2 * beam_group_size; ++i) {
    // Pop the element with largest value to "smemOutput" per iteration
    KVPair topKVPair =
        BlockReduceTopK(smemReduceBuffer.topk).Reduce(topKVPairPartial, argmax);
    if (tid == 0) {
      // const int index = beam_batch_id * vocab_size + section_start +
      const int index = section_start + topKVPair.key;
      reinterpret_cast<int *>(smemOutput)[i] = index;
      smemOutput[K + i] = topKVPair.value;
      smemLogProbs[topKVPair.key] =
          -MAX_T_VAL;  // pollute the value of the popped element
      threadToUpdate = topKVPair.key % THREADBLOCK_SIZE;
    }
    __syncthreads();

    if (tid == threadToUpdate && i < 2 * beam_group_size - 1) {
      // The thread popped the element need to update its topKVPairPartial
      // No need to do this in the last iteration
      topKVPairPartial.key = vocab_size - 1;
      topKVPairPartial.value = -MAX_T_VAL;
      for (int index = tid; index < valid_smem_length;
           index += THREADBLOCK_SIZE) {
        topKVPairPartial =
            argmax(topKVPairPartial, {index, smemLogProbs[index]});
      }
    }
  }

  // Do reduce_md among the top 2K elements in the smemOutput and write into
  // tail of smemOutput
  auto reduce_md_func = [](const MD &a, const MD &b) {
    return reduce_md_op(a, b);
  };
  MD total_md =
      BlockReduceMD(smemReduceBuffer.md).Reduce(partial_md, reduce_md_func);
  if (tid == 0) {
    smemOutput[2 * K] = total_md.d;
    smemOutput[2 * K + 1] = total_md.m;
  }
  __syncthreads();

  // Write the smemOutput into tmp_buffer
  float *local_temp_buffer =
      tmp_buffer + group_beam_batch_id * PACKED_TOP_KMD_SIZE * gridDim.y +
      blockIdx.y * PACKED_TOP_KMD_SIZE;
#pragma unroll
  for (int i = tid; i < PACKED_TOP_KMD_SIZE; i += THREADBLOCK_SIZE) {
    local_temp_buffer[i] = smemOutput[i];
  }
}

//<<<(batch_size * beam_group_size, voc_parts), 128>>>
template <typename T, int K, int THREADBLOCK_SIZE, int PACKED_TOP_KMD_SIZE>
__global__ void beam_search_softmax_topk_stage1(BeamSearchParams<T> params,
                                                const int beam_width,
                                                const int beam_group_idx,
                                                const int vocab_size,
                                                const bool fuse_softmax) {
  const int thread_id = threadIdx.x;
  const int beam_group_size = K / 2;
  const int batch_id = blockIdx.x / beam_group_size;
  const int beam_group_sub_idx = blockIdx.x % beam_group_size;
  const int beam_batch_id = batch_id * beam_width +
                            beam_group_idx * beam_group_size +
                            beam_group_sub_idx;

  const bool finish = params.beam_finished[beam_batch_id];
  const int seq_len = params.seq_lens[beam_batch_id];
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  if (blockIdx.y == 0 && thread_id == 0) {
    printf(
        "batch %d. beam_group_sub_idx %d. beam_batch_id %d. "
        "group_beam_batch_id %d. seq_len %d. \n",
        batch_id,
        beam_group_sub_idx,
        beam_batch_id,
        blockIdx.x,
        seq_len);
  }
#endif
  // for dybatch
  if (seq_len < 0 || finish) {
    return;
  }

  // 2 * K + 2
  __shared__ float buf_s[PACKED_TOP_KMD_SIZE];

  const T MAX_T_VAL = FLT_MAX;

  const int v_local = (vocab_size + gridDim.y - 1) / gridDim.y;
  const int section_start = v_local * blockIdx.y;
  int section_end = section_start + v_local;
  section_end = (section_end > vocab_size) ? vocab_size : section_end;

  T *logits = params.logits + beam_batch_id * vocab_size;
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX

  if (blockIdx.y == 0 && thread_id == 0) {
    printf("ID %d. section_start: %d. section_end: %d. logtis:%f\n",
           blockIdx.x,
           section_start,
           section_end,
           logits[0]);
  }
#endif
  if (fuse_softmax) {
    typedef cub::BlockReduce<TopKSoftMax<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKSoftMax<T, K> partial;
    for (int i = 0; i < K; ++i) {
      partial.topk.ids[i] = -1;
      partial.topk.vals[i] = -MAX_T_VAL;
    }
    partial.softmax_md.logit = -MAX_T_VAL;
    partial.softmax_md.score = 0.0F;

// process voc_parts
#pragma unroll 1
    for (int elem_id = section_start + thread_id; elem_id < section_end;
         elem_id += THREADBLOCK_SIZE) {
      T elem = logits[elem_id];
      DySoftMaxStruct new_elem{elem, 1.0F};
      partial.softmax_md = reduce_softmax_op(partial.softmax_md, new_elem);
      partial.topk.insert(elem, elem_id);
    }
    // === old_beam_search strategy ===
    // }

    // reduce voc_parts
    TopKSoftMax<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_softmax_op<T, K>);

    if (thread_id == 0) {
      for (int i = 0; i < K; i++) {
        reinterpret_cast<int *>(buf_s)[i] = total.topk.ids[i];
        buf_s[K + i] = total.topk.vals[i];
      }
      buf_s[2 * K] = total.softmax_md.score;
      buf_s[2 * K + 1] = total.softmax_md.logit;
    }
  } else {
    typedef cub::BlockReduce<TopK<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, K> partial;
    for (int i = 0; i < K; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -MAX_T_VAL;
    }

#pragma unroll 1
    for (int elem_id = section_start + thread_id; elem_id < section_end;
         elem_id += THREADBLOCK_SIZE) {
      T elem = logits[elem_id];
      partial.insert(elem, elem_id);
    }

    TopK<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, K>);

    if (thread_id == 0) {
      for (int i = 0; i < K; i++) {
        reinterpret_cast<int *>(buf_s)[i] = total.ids[i];
        buf_s[K + i] = total.vals[i];
      }
    }
  }
  __syncthreads();
  // write all the voc_parts results to tmp_buffer
  for (int elem_id = thread_id; elem_id < PACKED_TOP_KMD_SIZE;
       elem_id += THREADBLOCK_SIZE) {
    params.tmp_buffer[blockIdx.x * PACKED_TOP_KMD_SIZE * gridDim.y +
               blockIdx.y * PACKED_TOP_KMD_SIZE + elem_id] = buf_s[elem_id];
  }
}

template <typename T, int K, int THREADBLOCK_SIZE, bool IS_FAST_KERNEL>
__launch_bounds__(THREADBLOCK_SIZE) __global__
    void beam_search_softmax_topk_stage2_fast(
        int *__restrict tmp_ids,
        T *__restrict tmp_vals,
        float *__restrict tmp_buffer,
        const float *__restrict cum_scores,
        const bool *__restrict beam_finished,
        const int *__restrict seq_lens,
        const int beam_width,
        const int beam_group_idx,
        const int vocab_size,
        const int voc_parts) {
  constexpr int PACKED_TOP_KMD_SIZE = 2 * K + 2;
  constexpr int beam_group_size = K / 2;
  const int group_beam_batch_id = blockIdx.x;
  const int beam_group_sub_id = blockIdx.x % beam_group_size;
  const int batch_size = group_beam_batch_id / beam_group_size;
  const int beam_batch_id = batch_size * beam_width +
                            beam_group_idx * beam_group_size +
                            beam_group_sub_id;

  if (seq_lens[beam_batch_id] < 0 || beam_finished[beam_batch_id]) {
    return;
  }

  const int tid = threadIdx.x;
  T const MAX_T_VAL = FLT_MAX;

  using KVPair = cub::KeyValuePair<int, T>;
  using BlockReduceTopK = cub::BlockReduce<KVPair, THREADBLOCK_SIZE>;
  using BlockReduceMD = cub::BlockReduce<MD, THREADBLOCK_SIZE>;

  __shared__ KVPair buf_smem_kv[K];

  __shared__ union {
    typename BlockReduceTopK::TempStorage topk;
    typename BlockReduceMD::TempStorage md;
  } smemReduceBuffer;

  cub::ArgMax argmax;
  MD partial_md{-MAX_T_VAL, 0.0f};
  KVPair topKVPair{vocab_size - 1, -MAX_T_VAL};

  auto reduce_md_func = [](const MD &a, const MD &b) {
    return reduce_md_op(a, b);
  };

  // Load and unpack into registers through smem
  float *localTempBuffer =
      tmp_buffer + PACKED_TOP_KMD_SIZE * group_beam_batch_id * voc_parts;
  if constexpr (IS_FAST_KERNEL) {  // Use share memory instead of global memory
    extern __shared__ char smem[];
    float *smemVal = reinterpret_cast<float *>(smem);
    for (int idx = tid; idx < PACKED_TOP_KMD_SIZE * voc_parts; idx += THREADBLOCK_SIZE) {
      smemVal[idx] = localTempBuffer[idx];
    }
    localTempBuffer = smemVal;
    __syncthreads();
  }

  // Find the top 2K across all voc_parts
  for (int k = 0; k < K; ++k) {
    KVPair topKVPairPartial{vocab_size - 1, -MAX_T_VAL};
    // Only threads responsible for a chunk will do the computation
    if (tid < voc_parts) {
      for (int i = 0; i < K; ++i) {
        const int current_index = tid * PACKED_TOP_KMD_SIZE + i;
        T topValue = localTempBuffer[current_index + K];
        topKVPairPartial = argmax(topKVPairPartial, {current_index, topValue});
      }
    }

    KVPair topKVPair =
        BlockReduceTopK(smemReduceBuffer.topk).Reduce(topKVPairPartial, argmax);
    __syncthreads();

    if (tid == 0) {
      // Store kv pairs in shared mem buffer
      int temp_offset = topKVPair.key;
      int global_offset = reinterpret_cast<int *>(localTempBuffer)[temp_offset];
      topKVPair.key = global_offset;
      buf_smem_kv[k] = topKVPair;

      // Invalidate the maximum value within the chunk
      reinterpret_cast<int *>(localTempBuffer)[temp_offset] =
          vocab_size - 1;                             // id in share memory
      localTempBuffer[temp_offset + K] = -MAX_T_VAL;  // value in share memory
    }
    __syncthreads();
  }

  // Extract and reduce MD values across the chunks
  if (tid < voc_parts) {
    partial_md.d = localTempBuffer[tid * PACKED_TOP_KMD_SIZE + 2 * K];
    partial_md.m = localTempBuffer[tid * PACKED_TOP_KMD_SIZE + 2 * K + 1];
  }
  __syncthreads();

  MD total_md =
      BlockReduceMD(smemReduceBuffer.md).Reduce(partial_md, reduce_md_func);

  if (tid == 0) {
    float d_total_log = logf(total_md.d);

    for (int i = 0; i < K; ++i) {
      float val = (float)buf_smem_kv[i].value - total_md.m - d_total_log;
      tmp_ids[group_beam_batch_id * K + i] =
          buf_smem_kv[i].key;
      tmp_vals[group_beam_batch_id * K + i] =
          val + cum_scores[beam_batch_id];
    }
  }
}

#define BEAM_STAGE2_KERNEL(N_VOCAB_PART, IS_FAST_KERNEL)                     \
  do {                                                                       \
    if (IS_FAST_KERNEL && nShareMemory >= (48 << 10)) {                      \
      cudaFuncSetAttribute(                                                  \
          beam_search_softmax_topk_stage2_fast<T,                            \
                                               K,                            \
                                               N_VOCAB_PART,                 \
                                               IS_FAST_KERNEL>,              \
          cudaFuncAttributeMaxDynamicSharedMemorySize,                       \
          nShareMemory);                                                     \
    }                                                                        \
    beam_search_softmax_topk_stage2_fast<T, K, N_VOCAB_PART, IS_FAST_KERNEL> \
        <<<batch_size * beam_group_size,                                     \
           N_VOCAB_PART,                                                     \
           IS_FAST_KERNEL * nShareMemory,                                    \
           stream>>>(params.tmp_ids,                                         \
                     params.tmp_vals,                                        \
                     params.tmp_buffer,                                      \
                     params.cum_scores,                                      \
                     params.beam_finished,                                  \
                     params.seq_lens,                                        \
                     beam_width,                                             \
                     beam_group_idx,                                         \
                     vocab_size,                                             \
                     voc_parts);                                             \
  } while (0);                                                               \
  return;

template <typename T, int K>
__inline__ void beamSearchSoftmaxTopkStage2FastKernelLauncher(
    BeamSearchParams<T> &params,
    const int batch_size,
    const int beam_width,
    const int beam_group_idx,
    const int vocab_size,
    const int voc_parts,
    const int max_smem_per_block,
    cudaStream_t stream) {
  constexpr int beam_group_size = K / 2;
  size_t const nShareMemory = sizeof(float) * voc_parts * (2 * K + 2) +
                              sizeof(cub::KeyValuePair<int, T>) * K;
  if (nShareMemory < max_smem_per_block) {  // IS_FAST_KERNEL must be a
                                            // compilation-time constant
    if (voc_parts <= 32) {
      BEAM_STAGE2_KERNEL(32, true)
    }
    if (voc_parts <= 64) {
      BEAM_STAGE2_KERNEL(64, true)
    }
    BEAM_STAGE2_KERNEL(128, true)
    // No larger branch since voc_parts <= nMaxVocabPartForStage1FastKernel
  }
  BEAM_STAGE2_KERNEL(128, false)
}

template <typename T, int K, int THREADBLOCK_SIZE>
__global__ void beam_search_softmax_topk_stage2(BeamSearchParams<T> params,
                                                const int beam_width,
                                                const int beam_group_idx,
                                                const int voc_parts,
                                                const int packed_top_kmd_size,
                                                const bool fuse_softmax) {
  const int thread_id = threadIdx.x;
  const int beam_group_size = K / 2;
  const int batch_id = blockIdx.x / beam_group_size;
  const int beam_group_sub_idx = blockIdx.x % beam_group_size;
  // int vector_id = blockIdx.x;  // batch beam index.
  const int beam_batch_id = batch_id * beam_width +
                            beam_group_idx * beam_group_size +
                            beam_group_sub_idx;
  const int group_beam_batch_id = blockIdx.x;
  // const int vector_id = blockIdx.x;
  const int PACKED_TOP_KMD_SIZE = packed_top_kmd_size;
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  // printf("--stage2: group_beam_batch_id: %d beam_batch_id: %d\n",
  // group_beam_batch_id, beam_batch_id);
#endif
  // for dybatch
  const int seq_len = params.seq_lens[beam_batch_id];
  const bool finish = params.beam_finished[beam_batch_id];

  int *tmp_ids = params.tmp_ids + group_beam_batch_id * K;
  float *tmp_vals = params.tmp_vals + group_beam_batch_id * K;
  float *tmp_buffer = params.tmp_buffer;

  const T *cum_scores = params.cum_scores + beam_batch_id;
  if (seq_len < 0 || finish) {
    return;
  }
  const T MAX_T_VAL = FLT_MAX;

  extern __shared__ char buf_s_[];
  float *buf_s = reinterpret_cast<float *>(buf_s_);
  // 当前 batch beam 的所有 voc
  tmp_buffer += group_beam_batch_id * PACKED_TOP_KMD_SIZE * voc_parts;

  if (fuse_softmax) {
    typedef cub::BlockReduce<TopKSoftMax<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopKSoftMax<T, K> partial;
    for (int i = 0; i < K; ++i) {
      partial.topk.ids[i] = -1;
      partial.topk.vals[i] = -MAX_T_VAL;
    }
    partial.softmax_md.logit = -MAX_T_VAL;
    partial.softmax_md.score = 0.0F;

    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * voc_parts;
         idx += THREADBLOCK_SIZE) {
      buf_s[idx] = tmp_buffer[idx];
    }
    __syncthreads();

    if (threadIdx.x < voc_parts) {
      float *b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
      for (int i = 0; i < K; i++) {
        partial.topk.ids[i] = reinterpret_cast<int *>(b_s)[i];
        partial.topk.vals[i] = b_s[K + i];
      }
      partial.softmax_md.score = b_s[2 * K];
      partial.softmax_md.logit = b_s[2 * K + 1];
    }
    __syncthreads();

    TopKSoftMax<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_softmax_op<T, K>);

    if (thread_id == 0) {
      // tmp_ids += group_beam_batch_id * K;
      // tmp_vals += group_beam_batch_id * K;

      float d_total_log = logf(total.softmax_md.score);
      for (int i = 0; i < K; ++i) {
        // float val = expf((float)total.topk.vals[i] - total.softmax_md.logit -
        // d_total_log);
        float val = total.topk.vals[i] - total.softmax_md.logit - d_total_log;
        tmp_ids[i] = total.topk.ids[i];
        tmp_vals[i] = val + params.cum_scores[beam_batch_id];
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
        printf(
            "group_beam_batch_id: %d, vals: %f, logit: %f, d_total_log: %f,id: %d, val: "
            "%f, cum_log_probs: %f, res: %f\n",
            group_beam_batch_id,
            total.topk.vals[i],
            total.softmax_md.logit,
            d_total_log,
            tmp_ids[i],
            val,
            params.cum_scores[beam_batch_id],
            tmp_vals[i]);
#endif
      }
    }
  } else {
    typedef cub::BlockReduce<TopK<T, K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<T, K> partial;
    for (int i = 0; i < K; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -MAX_T_VAL;
    }

    for (int idx = thread_id; idx < PACKED_TOP_KMD_SIZE * voc_parts;
         idx += THREADBLOCK_SIZE) {
      buf_s[idx] = tmp_buffer[idx];
    }
    __syncthreads();

    if (threadIdx.x < voc_parts) {
      float *b_s = buf_s + thread_id * PACKED_TOP_KMD_SIZE;
      for (int i = 0; i < K; i++) {
        partial.ids[i] = reinterpret_cast<int *>(b_s)[i];
        partial.vals[i] = b_s[K + i];
      }
    }
    __syncthreads();

    TopK<T, K> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, K>);

    if (thread_id == 0) {
      tmp_ids += group_beam_batch_id * K;
      tmp_vals += group_beam_batch_id * K;

      for (int i = 0; i < K; ++i) {
        float val = total.vals[i];
        tmp_ids[i] = total.ids[i];
        tmp_vals[i] = val + params.cum_scores[beam_batch_id];
      }
    }
  }
}

template <typename T, int K>
void invokeBeamSearchSoftmaxTopKStage2(BeamSearchParams<T> &params,
                                       const int batch_size,
                                       const int beam_width,
                                       const int beam_group_idx,
                                       const int voc_parts,
                                       const int packed_top_kmd_size,
                                       const bool fuse_softmax,
                                       gpuStream_t stream) {
  int smem_stage2_size = voc_parts * packed_top_kmd_size * sizeof(float);
  const int beam_group_size = K / 2;
  if (voc_parts <= 32) {
    beam_search_softmax_topk_stage2<T, K, 32>
        <<<batch_size * beam_group_size, 32, smem_stage2_size, stream>>>(
            params,
            beam_width,
            beam_group_idx,
            voc_parts,
            packed_top_kmd_size,
            fuse_softmax);
    return;
  }
  if (voc_parts <= 64) {
    beam_search_softmax_topk_stage2<T, K, 64>
        <<<batch_size * beam_group_size, 64, smem_stage2_size, stream>>>(
            params,
            beam_width,
            beam_group_idx,
            voc_parts,
            packed_top_kmd_size,
            fuse_softmax);
    return;
  }
  if (voc_parts <= 128) {
    beam_search_softmax_topk_stage2<T, K, 128>
        <<<batch_size * beam_group_size, 128, smem_stage2_size, stream>>>(
            params,
            beam_width,
            beam_group_idx,
            voc_parts,
            packed_top_kmd_size,
            fuse_softmax);
    return;
  }
  if (voc_parts <= 256) {
    beam_search_softmax_topk_stage2<T, K, 256>
        <<<batch_size * beam_group_size, 256, smem_stage2_size, stream>>>(
            params,
            beam_width,
            beam_group_idx,
            voc_parts,
            packed_top_kmd_size,
            fuse_softmax);
    return;
  }
}


template <typename T, int K>
__global__ void update_beam_finished_early_stop(const T *beam_hyps_score_out,
                                                bool *beam_finished) {
  if (threadIdx.x == 0) {
    int batch_idx = blockIdx.x;

    const T *cur_beam_hyps_score = beam_hyps_score_out + batch_idx * K;
    bool *cur_beam_finished = beam_finished + batch_idx * K;
    if (cur_beam_hyps_score[K - 1] > -1e8) {
      for (int i = 0; i < K; i++) {
        cur_beam_finished[i] = true;
      }
    }
  }
}

// <<<batch_size>>>
template <typename T, int K, int THREADBLOCK_SIZE, bool GROUP>
__global__ void batch_topk(BeamSearchParams<T> params,
                           const int beam_width,
                           const int beam_group_idx,
                           const int dec_stride) {
  const bool early_stop = params.early_stop;
  const int thread_id = threadIdx.x;
  const int batch_id = blockIdx.x;
  // int block_id = blockIdx.x;  // bs
  const int beam_group_size = K / 2;
  const int beam_group_start_id =
      batch_id * beam_width + beam_group_idx * beam_group_size;

  bool *beam_finished = params.beam_finished + beam_group_start_id;
  const int *step_ids = params.step_ids + beam_group_start_id;
  int *next_tokens = params.next_tokens + beam_group_start_id;
  float *cum_scores_out = params.cum_scores_out + beam_group_start_id;
  int *parent_ids = params.parent_ids + beam_group_start_id;
  float *beam_hyps_score_out = params.beam_hyps_score_out + beam_group_start_id;

  const bool finish = beam_finished[0];
  const int step_id = step_ids[0];
  const int seq_len = params.seq_lens[beam_group_start_id];
  const int max_dec_len = params.max_dec_lens[beam_group_start_id];

  const bool last_dec_step = (step_id + 1 == max_dec_len);
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  if (finish && thread_id == 0) {
    printf("batch_topk: batch %d finish \n", beam_group_start_id);
  }
#endif

  if (thread_id == 0 && seq_len > 0 && !finish) {
    TopK<T, K> partial;
    BeamHypothesesTopK<T, K / 2> beam_hyps;

    beam_hyps.init(params.beam_hyps_out + beam_group_start_id * dec_stride,
                   params.beam_hyps_score_out + beam_group_start_id,
                   dec_stride);

    for (int i = 0; i < K; ++i) {
      partial.ids[i] = -1;
      partial.vals[i] = -FLT_MAX;
      partial.parent_ids[i] = -1;
    }
    int index = batch_id * beam_group_size * K;
    if (step_id == 0) {
      for (int i = 0; i < K; i++) {
        float score_now = apply_length_penalty(
            params.tmp_vals[index + i], step_id + 1, params.length_penalty[batch_id]);
        if (!GROUP) {
          score_now -= params.diversity_penalty[batch_id] * static_cast<float>(i + 1);
        }
        partial.insert((T)score_now, params.tmp_ids[index + i], i / K);
      }
    } else {
      for (int i = 0; i < beam_group_size * K; i++) {
        float score_now = apply_length_penalty(
            params.tmp_vals[index + i], step_id + 1, params.length_penalty[batch_id]);
        if (!GROUP) {
          score_now -= params.diversity_penalty[batch_id] * static_cast<float>(i % K + 1);
        }
        partial.insert((T)score_now, params.tmp_ids[index + i], i / K);
      }
    }
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
    for (int i = 0; i < K; ++i) {
      printf("Batch %d. TopK: %d. id:%d. val: %f. parent: %d \n",batch_id, i,
      partial.ids[i], partial.vals[i], partial.parent_ids[i]);
    }
#endif
    if (partial.vals[0] < beam_hyps.hyps[beam_group_size - 1].score) {
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
      printf("batch %d best score %f < worst_hyp_score. stop\n",
             batch_id,
             params.cum_scores[index],
             beam_hyps.hyps[beam_group_size - 1].score);
#endif
      for (int i = 0; i < beam_group_size; i++) {
        beam_finished[i] = true;
      }
      return;
    }

    int next_step_num = 0;
    for (int i = 0; i < K && next_step_num < beam_group_size; i++) {
      int parent_id = partial.parent_ids[i];
      if (is_in_end(partial.ids[i], params.end_ids, params.end_ids_len) ||
          last_dec_step) {
        if (i < beam_group_size &&
            partial.vals[i] > beam_hyps.get_worst_score()) {
          const int *beam_cache_id = params.beam_cache_ids +
                                     beam_group_start_id * dec_stride +
                                     parent_id * dec_stride;
          beam_hyps.insert(beam_cache_id,
                           step_id,
                           last_dec_step ? params.end_ids[0] : partial.ids[i],
                           partial.vals[i]);
        }

        if (early_stop && beam_hyps.get_worst_score() > -1e8) {
          // stop
          for (int i = 0; i < beam_group_size; i++) {
            beam_finished[i] = true;
          }
          return;
        }
      } else {
        next_tokens[next_step_num] = partial.ids[i];
        cum_scores_out[next_step_num] = partial.vals[i];
        parent_ids[next_step_num] = parent_id;
        next_step_num += 1;
      }
    }  // for
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
    for (int i = 0; i < K / 2; i++) {
      printf("buf: %d-%d. id:%d. val: %f. parent: %d \n",
             batch_id,
             i,
             next_tokens[i],
             cum_scores_out[i],
             parent_ids[i]);
    }
#endif
    for (int i = 0; i < beam_group_size; i++) {
      beam_hyps_score_out[i] = beam_hyps.hyps[i].score;
    }

    if (last_dec_step) {
      for (int i = 0; i < beam_group_size; i++) {
        beam_finished[i] = true;
      }
    }
  }  // if (thread_id == 0)
}

template <typename T, typename Context, int K, bool GROUP>
void invokeTopKSoftMaxLauncher(const Context &dev_ctx,
                               BeamSearchParams<T> &params,
                               int beam_group_idx,
                               gpuStream_t stream) {
  const int batch_size = params.batch_size;
  const int beam_width = params.beam_width;
  const int beam_group_size = K / 2;
  const int vocab_size = params.vocab_size;
  const bool fuse_softmax = params.fuse_softmax;
  const int voc_parts = params.voc_parts;
  constexpr int dev_id = 0;

  // only in group_beam_search
  if (beam_width > beam_group_size && beam_group_idx != 0) {
    apply_group_diversity_penalty<T, K>
        <<<batch_size, beam_group_size, 0, stream>>>(
            params, batch_size, beam_width, beam_group_idx, vocab_size);
  }

  // == Step1 == : stage1
  if (params.use_fast_kernel) {
    constexpr int block_size =
      (K < 16) ? ((K < 8) ? kBlockSizeForSmallBeamWidth : 128) : 64;
    const int vocab_chunk_size = (vocab_size + voc_parts - 1) / voc_parts;
    const int dyn_smem_size = sizeof(T) * vocab_chunk_size;
    VLOG(1) << "Stage1 kernel dyn_smem_size: " << dyn_smem_size;
    if (dyn_smem_size >= (48 << 10)) {
      cudaFuncSetAttribute(beam_search_softmax_topk_stage1_fast<T, K, block_size>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            dyn_smem_size);
    }
    VLOG(1) << "voc_parts: " << voc_parts;

    dim3 grid(batch_size * beam_group_size, voc_parts);
    beam_search_softmax_topk_stage1_fast<T, K, block_size>
        <<<grid, block_size, dyn_smem_size, stream>>>(params.logits,
                                                      params.tmp_buffer,
                                                      params.end_ids,
                                                      params.beam_finished,
                                                      params.seq_lens,
                                                      beam_width,
                                                      beam_group_idx,
                                                      vocab_size,
                                                      vocab_chunk_size);
  } else {
    constexpr int block_size = 128;
    VLOG(1) << "Old Stage1 kernel";
    dim3 grid(batch_size * beam_group_size, voc_parts);
    cudaFuncSetAttribute(
        beam_search_softmax_topk_stage1<float, K, block_size, 2 * K + 2>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxL1);
    if (fuse_softmax) {
#ifdef PADDLE_WITH_CUDA
      cudaFuncSetAttribute(
          beam_search_softmax_topk_stage1<T, K, block_size, 2 * K + 2>,
          cudaFuncAttributePreferredSharedMemoryCarveout,
          cudaSharedmemCarveoutMaxL1);
#else
      // cudaSharedmemCarveoutMaxL1 equal to 0
      hipFuncSetAttribute(
          reinterpret_cast<void *>(
              beam_search_softmax_topk_stage1<T, K, block_size, 2 * K + 2>),
          hipFuncAttributePreferredSharedMemoryCarveout,
          0);
#endif
      // （bs, bm, voc_parts, 2 * K + 2）
      beam_search_softmax_topk_stage1<T, K, block_size, 2 * K + 2>
          <<<grid, block_size, 0, stream>>>(params,
                                            beam_width,
                                            beam_group_idx,
                                            vocab_size,
                                            fuse_softmax);
    } else {
#ifdef PADDLE_WITH_CUDA
      cudaFuncSetAttribute(
          beam_search_softmax_topk_stage1<T, K, block_size, 2 * K>,
          cudaFuncAttributePreferredSharedMemoryCarveout,
          cudaSharedmemCarveoutMaxL1);
#else
      // cudaSharedmemCarveoutMaxL1 equal to 0
      hipFuncSetAttribute(
          reinterpret_cast<void *>(
              beam_search_softmax_topk_stage1<T, K, block_size, 2 * K>),
          hipFuncAttributePreferredSharedMemoryCarveout,
          0);
#endif
      // （bs, bm, voc_parts, 2 * K）
      beam_search_softmax_topk_stage1<T, K, block_size, 2 * K>
          <<<grid, block_size, 0, stream>>>(params,
                                            beam_width,
                                            beam_group_idx,
                                            vocab_size,
                                            fuse_softmax);
    }
  }

  // Reserved for debug
  // invokeBeamSearchSoftmaxTopKStage2<T, K>(params,
  //                                         batch_size,
  //                                         beam_width,
  //                                         beam_group_idx,
  //                                         voc_parts,
  //                                         packed_top_kmd_size,
  //                                         fuse_softmax,
  //                                         stream);
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  printf("======== num %dth for loop before stage2 =======\n", beam_group_idx);
  int total_ele = batch_size * beam_width;  // hard code here
  int parent_ids_arr[total_ele];
  cudaMemcpy(parent_ids_arr,
             params.parent_ids,
             total_ele * sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("parent_ids_arr total: \n");
  for (int i = 0; i < total_ele; i++) {
    printf("%d-%f. ", parent_ids_arr[i], (float*)reinterpret_cast<float*>(parent_ids_arr+i));
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
  int packed_top_kmd_size = 2 * K;
  if (fuse_softmax) {
    packed_top_kmd_size += 2;
  }
  const int tmp_buffer_size =
      batch_size * beam_group_size * voc_parts * packed_top_kmd_size;

  VLOG(0) << "tmp_buffer_size invoke: " << tmp_buffer_size;
  float* tmp_buffer_cpu = (float*)malloc(tmp_buffer_size * sizeof(float));
  printf("point3 %p\n", params.tmp_buffer);
  cudaMemcpy(tmp_buffer_cpu, params.tmp_buffer, tmp_buffer_size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < batch_size; i++) {
    printf("--- batch %d ---\n", i);
    for (int j = 0; j < beam_group_size; j++) {
      printf("  -- sub_group_id %d --\n", j);
      for (int k = 0; k < voc_parts; k++) {
        printf("    - voc part id - %d - \n    ", k);
        for (int elem_id = 0; elem_id < K; elem_id++) {
          printf("%dth idx:%d.  ", elem_id, *reinterpret_cast<int*>(tmp_buffer_cpu + i * beam_group_size * voc_parts * packed_top_kmd_size + 
                  j * voc_parts * packed_top_kmd_size + k * packed_top_kmd_size + elem_id));
        }
        printf("\n    ");
        for (int elem_id = K ; elem_id < packed_top_kmd_size; elem_id++) {
          printf("%dth val:%f.  ", elem_id, tmp_buffer_cpu[i * beam_group_size * voc_parts * packed_top_kmd_size + 
                  j * voc_parts * packed_top_kmd_size + k * packed_top_kmd_size + elem_id]);

        }
        printf("\n");
      }
    }
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#endif
  beamSearchSoftmaxTopkStage2FastKernelLauncher<T, K>(params,
                                                      batch_size,
                                                      beam_width,
                                                      beam_group_idx,
                                                      vocab_size,
                                                      voc_parts,
                                                      params.max_smem_per_block,
                                                      stream);

  batch_topk<T, K, 32, GROUP><<<batch_size, 32, 0, stream>>>(
      params, beam_width, beam_group_idx, params.dec_stride);
  // === old_beam_search strategy ===
  // }

#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  printf("======== num %dth for loop =======\n", beam_group_idx);
  // int total_ele = batch_size * beam_width;  // hard code here
  // int parent_ids_arr[total_ele];
  cudaMemcpy(parent_ids_arr,
             params.parent_ids,
             total_ele * sizeof(int),
             cudaMemcpyDeviceToHost);
  printf("parent_ids_arr total: \n");
  for (int i = 0; i < total_ele; i++) {
    printf("%d. ", parent_ids_arr[i]);
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

  bool beam_finished_arr[total_ele];
  cudaMemcpy(beam_finished_arr,
             params.beam_finished,
             sizeof(bool) * total_ele,
             cudaMemcpyDeviceToHost);
  printf("beam_finished total: \n");
  for (int i = 0; i < total_ele; i++) {
    printf("%d. ", beam_finished_arr[i]);
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());

  int next_tokens_arr[total_ele];
  cudaMemcpy(next_tokens_arr,
             params.next_tokens,
             sizeof(int) * total_ele,
             cudaMemcpyDeviceToHost);
  printf("next_tokens total: \n");
  for (int i = 0; i < total_ele; i++) {
    printf("%d. ", next_tokens_arr[i]);
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#endif
}

template <typename T, typename Context, bool GROUP>
void invokeTopkSoftMax(const Context &dev_ctx,
                       BeamSearchParams<T> &params,
                       int beam_group_idx,
                       gpuStream_t stream) {
  switch (params.beam_group_size) {
    CASE_K(1);
    CASE_K(2);
    CASE_K(3);
    CASE_K(4);
    CASE_K(5);
    CASE_K(6);
    CASE_K(7);
    CASE_K(8);
    CASE_K(9);
    CASE_K(10);
    CASE_K(11);
    CASE_K(12);
    CASE_K(13);
    CASE_K(14);
    CASE_K(15);
    CASE_K(16);
    default:
      PADDLE_THROW(errors::InvalidArgument(
            "Beam_group_size/Beam_width must <= 16, but get %d",
            params.beam_group_size));
  }
}


template<typename T, int K>
void ComputeVocParts(BeamSearchParams<T> &params) {
  int dev_id = 0;
  const int block_size =
      (K < 16) ? ((K < 8) ? kBlockSizeForSmallBeamWidth : 128) : 64;
  int max_active_blocks = -1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks,
      beam_search_softmax_topk_stage1_fast<float, K, block_size>,
      block_size,
      0);

  int max_smem_per_sm = -1;
  int max_smem_per_block = -1;
  cudaDeviceGetAttribute(
      &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id);
  cudaDeviceGetAttribute(
      &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(
      &attr, beam_search_softmax_topk_stage1_fast<float, K, block_size>);
  const int static_smem = attr.sharedSizeBytes;
  const int max_dyn_smem_per_block = max_smem_per_block - static_smem;

  if (sizeof(T) * params.vocab_size >
      max_dyn_smem_per_block * kMaxVocabPartForStage1FastKernel) {
    VLOG(1) << "Vocab size is too large. Back to old kernel.";
  }

  VLOG(1) << "max_active_blocks: " << max_active_blocks;
  const int driver_smem_per_block = max_smem_per_sm - max_smem_per_block;
  const int extra_smem = driver_smem_per_block + static_smem;
  VLOG(1) << "max_smem_per_sm: " << max_smem_per_sm
          << ". max_smem_per_block: " << max_smem_per_block
          << ". extra_smem: " << extra_smem;
  int voc_parts = kMaxVocabPartForStage1FastKernel + 1;
  VLOG(1) << "Start compute voc_parts";
  for (int n_block = max_active_blocks - 1; n_block > 0 && voc_parts > kMaxVocabPartForStage1FastKernel; --n_block) {
    int dyn_smem_size = max_smem_per_sm / n_block - extra_smem;
    dyn_smem_size -= dyn_smem_size % sizeof(T);
    voc_parts = ceilDiv(sizeof(T) * params.vocab_size, dyn_smem_size);
    VLOG(1) << "n_block: " << n_block << ". dyn_smem_size:" << dyn_smem_size
            << ". voc_parts: " << voc_parts;
  }

  if (!params.fuse_softmax || voc_parts > kMaxVocabPartForStage1FastKernel) {
    params.use_fast_kernel = false;
    VLOG(1) << "Vocab size is too big for shared-memory. Falling back to the old algorithm";
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
    const int max_act_blocks_per_sm = 4;
    const int max_act_blocks_per_wave = sm_count * max_act_blocks_per_sm;
    const int gridx = params.batch_size * K / 2;
    const int max_part_num = (max_act_blocks_per_wave + gridx - 1) / gridx;
    voc_parts = min(128, max_part_num);
  }
  params.voc_parts = voc_parts;
  params.max_smem_per_block = max_smem_per_block;
  VLOG(1) << "BeamSearch Pre-compute. voc_parts: " << params.voc_parts << ". use_fast_kernel: " << params.use_fast_kernel;
}

template<typename T>
void DispatchComputeVocParts(BeamSearchParams<T> &params){
  switch (params.beam_group_size) {
    DISPATCH_COMPUTE_PARTS_K(1);
    DISPATCH_COMPUTE_PARTS_K(2);
    DISPATCH_COMPUTE_PARTS_K(3);
    DISPATCH_COMPUTE_PARTS_K(4);
    DISPATCH_COMPUTE_PARTS_K(5);
    DISPATCH_COMPUTE_PARTS_K(6);
    DISPATCH_COMPUTE_PARTS_K(7);
    DISPATCH_COMPUTE_PARTS_K(8);
    DISPATCH_COMPUTE_PARTS_K(9);
    DISPATCH_COMPUTE_PARTS_K(10);
    DISPATCH_COMPUTE_PARTS_K(11);
    DISPATCH_COMPUTE_PARTS_K(12);
    DISPATCH_COMPUTE_PARTS_K(13);
    DISPATCH_COMPUTE_PARTS_K(14);
    DISPATCH_COMPUTE_PARTS_K(15);
    DISPATCH_COMPUTE_PARTS_K(16);
    default:
      PADDLE_THROW(errors::InvalidArgument(
            "Beam_group_size/Beam_width must <= 16, but get %d",
            params.beam_group_size));
  }
}

template <typename T>
__global__ void update_beam_search_params_kernel(BeamSearchParams<T> params) {
  int bb_id = blockIdx.y;
  int time_step = threadIdx.x + blockIdx.x * blockDim.x;

  const bool finished = params.beam_finished[bb_id];
  const int seq_len = params.seq_lens[bb_id];

  if (bb_id >= params.beam_width * params.batch_size) {
    return;
  }

  if (finished || seq_len < 0) {
    return;
  }

  const int beam_group_size = params.beam_group_size;
  const int max_seq_len = params.max_seq_len;
  const int dec_stride = params.dec_stride;

  const int batch_group_id = bb_id / beam_group_size;

  const int max_dec_len = params.max_dec_lens[bb_id];
  const int src_beam = params.parent_ids[bb_id];
  const int step = params.step_ids[bb_id];

  const int *block_tables = params.block_tables;
  int *block_tables_out = params.block_tables_out;
  const int *cache_ids = params.beam_cache_ids;
  int *cache_ids_out = params.cache_ids_out;
  const int *next_tokens = params.next_tokens;

  const int beam_group_sub_id = bb_id % beam_group_size;
  // const int src_bb_id = batch_group_id * beam_group_size + src_beam;

  if (time_step < min(max_seq_len, seq_len + 1)) {
    const uint block_tables_tgt_offset =
        batch_group_id * beam_group_size * max_seq_len +
        beam_group_sub_id * max_seq_len + time_step;
    const uint block_tables_src_offset =
        batch_group_id * beam_group_size * max_seq_len +
        src_beam * max_seq_len + time_step;
    block_tables_out[block_tables_tgt_offset] =
        block_tables[block_tables_src_offset];
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  printf("block_table. src_beam %d. time_step. %d. bid:%d, subID:%d, tgt_offset:%d, src_offset:%d, val: %d \n", src_beam, time_step, bb_id / params.beam_width, beam_group_sub_id, block_tables_tgt_offset, block_tables_src_offset, block_tables[block_tables_src_offset]);
#endif
    if (time_step < min(step + 1, max_dec_len)) {
      const uint cache_ids_tgt_offset =
          batch_group_id * beam_group_size * dec_stride +
          beam_group_sub_id * dec_stride + time_step;
      const uint cache_ids_src_offset =
          batch_group_id * beam_group_size * dec_stride +
          src_beam * dec_stride + time_step;
      cache_ids_out[cache_ids_tgt_offset] =
          (time_step == step) ? next_tokens[bb_id]
                              : cache_ids[cache_ids_src_offset];
#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  printf("cache_ids. src_beam %d. time_step. %d, bid:%d, subID:%d, tgt_offset:%d, src_offset:%d, val:%d\n", src_beam, time_step, bb_id / params.beam_width, beam_group_sub_id, cache_ids_tgt_offset, cache_ids_src_offset, cache_ids_out[cache_ids_tgt_offset]);
#endif
    }
  }
}

template <typename T>
__global__ void update_stop_flags(BeamSearchParams<T> params) {
  int bid = blockIdx.x;
  const int beam_width = params.beam_width;
  const int beam_group_size = params.beam_group_size;
  const bool* beam_finished = params.beam_finished + beam_width * bid;
  bool* stop_flags = params.stop_flags + beam_width * bid;
  bool finished = true;
  if (threadIdx.x == 0 && !stop_flags[0]) {
#pragma unroll
    for (int i = 0; i < beam_width; i += beam_group_size) {
      finished &= beam_finished[i];
    }
    if (finished) {
#pragma unroll
      for (int i = 0; i < beam_width; i++) {
        stop_flags[i] = true;
      }
    }
  }
}

template <typename T>
void updateBeamSearchParams(BeamSearchParams<T> &params, cudaStream_t stream) {
  const dim3 block(32);
  const dim3 grid((params.max_seq_len + block.x - 1) / block.x,
                  params.batch_size * params.beam_width);

  update_beam_search_params_kernel<<<grid, block, 0, stream>>>(params);

  const dim3 grid_2(params.batch_size);
  update_stop_flags<<<grid_2, 1, 0, stream>>>(params);
}

/*****
liuzichang01(Note): In order to adapt to the model structure of 5.2 without
adding while op and without affecting the speed. Use a 'fake inplace' method
here. Not elegant but useful ︸_︸.
*****/
template <typename T, typename Context>
void BeamSearchSoftmaxKernel(const Context &dev_ctx,
                             const DenseTensor &logits,
                             const DenseTensor &seq_lens,  
                             const DenseTensor &stop_flags,       // inplace
                             const DenseTensor &end_ids,
                             const DenseTensor &step_ids,  
                             const DenseTensor &max_dec_lens,
                             const DenseTensor &block_tables,     // inplace
                             const DenseTensor &cum_scores,       // inplace
                             const DenseTensor &beam_cache_ids,   // inplace
                             const DenseTensor &beam_hyps,        // inplace
                             const DenseTensor &beam_hyps_score,  // inplace
                             const DenseTensor &beam_finished,    // inplace
                             const DenseTensor &beam_width,
                             const DenseTensor &beam_group_num,
                             const DenseTensor &length_penalty,
                             const DenseTensor &diversity_penalty,
                             bool fuse_softmax,
                             bool early_stop,
                             DenseTensor *next_tokens,
                             DenseTensor *parent_ids) {
  // PADDLE_ENFORCE_EQ(beam_width % beam_group_num, 
  //                   0, 
  //                   platform::errors::InvalidArgument(
  //                     "beam_width must be divisible by beam_group_num."
  //                   ));
  
  const auto &logits_dims = logits.dims();

  int beam_width_scalar;
  cudaMemcpyAsync(&beam_width_scalar, beam_width.data<int>(), sizeof(int), cudaMemcpyDeviceToHost, dev_ctx.stream());

  int beam_group_num_scalar;
  cudaMemcpyAsync(&beam_group_num_scalar, beam_group_num.data<int>(), sizeof(int), cudaMemcpyDeviceToHost, dev_ctx.stream());

  int beam_batch_size = logits_dims[0];
  int batch_size = beam_batch_size / beam_width_scalar;
  int vocab_size = logits_dims[1];
  const int max_seq_len = block_tables.dims()[1];
  // liuzichang: In some cases, the length of Tensor is longer than max_dec_lens
  const int dec_stride = beam_hyps.dims()[1];
  const int end_ids_len = end_ids.dims()[0];
  const int beam_group_size = beam_width_scalar / beam_group_num_scalar;

#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  VLOG(2) << "beam_width: " << beam_width_scalar << ", beam_group_num: " << beam_group_num_scalar;
  VLOG(2) << "bsf-input -- logits: " << logits;
  VLOG(2) << "bsf-input -- cum_scores: " << cum_scores;
  VLOG(2) << "bsf-input -- seq_lens: " << seq_lens;
  VLOG(2) << "bsf-input -- beam_finished: " << beam_finished;
  VLOG(2) << "bsf-input -- end_ids: " << end_ids;
  VLOG(2) << "bsf-input -- step_ids: " << step_ids;
  VLOG(2) << "bsf-input -- beam_cache_ids: " << beam_cache_ids;
  VLOG(2) << "bsf-input -- block_tables: " << block_tables;
  VLOG(2) << "bsf-input -- beam_hyps: " << beam_hyps;
  VLOG(2) << "bsf-input -- beam_hyps_score: " << beam_hyps_score;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#endif

  dev_ctx.template Alloc<int>(next_tokens);
  dev_ctx.template Alloc<int>(parent_ids);
  cudaMemset(parent_ids->data<int>(), 0, beam_batch_size * sizeof(int));

  DenseTensor cum_scores_ori;
  cum_scores_ori.Resize(cum_scores.dims());
  dev_ctx.template Alloc<T>(&cum_scores_ori);

  DenseTensor beam_cache_ids_ori;
  beam_cache_ids_ori.Resize(beam_cache_ids.dims());
  dev_ctx.template Alloc<int>(&beam_cache_ids_ori);

  DenseTensor block_tables_ori;
  block_tables_ori.Resize(block_tables.dims());
  dev_ctx.template Alloc<int>(&block_tables_ori);

  phi::Copy(
      dev_ctx, beam_cache_ids, dev_ctx.GetPlace(), false, &beam_cache_ids_ori);
  phi::Copy(dev_ctx, cum_scores, dev_ctx.GetPlace(), false, &cum_scores_ori);
  phi::Copy(
      dev_ctx, block_tables, dev_ctx.GetPlace(), false, &block_tables_ori);

  const int tmp_size = batch_size * beam_group_size * beam_group_size * 2;
  DenseTensor tmp_topk_id;
  tmp_topk_id.Resize(phi::make_ddim({tmp_size}));
  dev_ctx.template Alloc<int>(&tmp_topk_id);

  DenseTensor tmp_topk_val;
  tmp_topk_val.Resize(phi::make_ddim({tmp_size}));
  dev_ctx.template Alloc<T>(&tmp_topk_val);

  BeamSearchParams<T> params;
  params.batch_size = batch_size;
  params.beam_width = beam_width_scalar;
  params.beam_group_size = beam_group_size;

  params.vocab_size = vocab_size;
  params.dec_stride = dec_stride;
  params.max_seq_len = max_seq_len;
  params.end_ids_len = end_ids_len;

  params.fuse_softmax = fuse_softmax;
  params.early_stop = early_stop;

  // Only Read
  params.step_ids = step_ids.data<int>();
  params.seq_lens = seq_lens.data<int>();
  params.max_dec_lens = max_dec_lens.data<int>();
  params.end_ids = end_ids.data<int>();
  params.length_penalty = length_penalty.data<float>();
  params.diversity_penalty = diversity_penalty.data<float>();

  params.cum_scores = cum_scores_ori.data<T>();
  params.block_tables = block_tables_ori.data<int>();
  params.beam_cache_ids = beam_cache_ids_ori.data<int>();

  // Write
  params.logits = const_cast<T *>(logits.data<T>());
  params.cache_ids_out = const_cast<int *>(beam_cache_ids.data<int>());
  params.block_tables_out = const_cast<int *>(block_tables.data<int>());
  params.cum_scores_out = const_cast<T *>(cum_scores.data<T>());
  params.beam_hyps_out = const_cast<int *>(beam_hyps.data<int>());
  params.beam_hyps_score_out = const_cast<T *>(beam_hyps_score.data<T>());
  params.beam_finished = const_cast<bool *>(beam_finished.data<bool>());
  params.stop_flags = const_cast<bool *>(stop_flags.data<bool>());

  params.next_tokens = next_tokens->data<int>();
  params.parent_ids = parent_ids->data<int>();

  params.tmp_ids = tmp_topk_id.data<int>();
  params.tmp_vals = tmp_topk_val.data<T>();

  DispatchComputeVocParts<T>(params);
  // allocate workspace 
  const int tmp_id_val_size = batch_size * beam_group_size * beam_group_size * 2;
  const int packed_top_kmd_size = fuse_softmax ? 2 * 2 * beam_group_size + 2 : 2 * 2 * beam_group_size;
  const int tmp_stage1_to_stage2_size =
      batch_size * beam_group_size * params.voc_parts * packed_top_kmd_size;

  const int workspace_size = tmp_id_val_size * 2 + tmp_stage1_to_stage2_size;
  DenseTensor wsp_buffer_tensor;
  wsp_buffer_tensor.Resize(phi::make_ddim({workspace_size}));
  dev_ctx.template Alloc<float>(&wsp_buffer_tensor);
  params.tmp_ids = reinterpret_cast<int*>(wsp_buffer_tensor.data<float>());
  params.tmp_vals = wsp_buffer_tensor.data<float>() + tmp_id_val_size;
  params.tmp_buffer = wsp_buffer_tensor.data<float>() + 2 * tmp_id_val_size;
  VLOG(2) << "tmp_id_val_size: " << tmp_id_val_size << ". tmp_stage1_to_stage2_size: " << tmp_stage1_to_stage2_size;

  for (int beam_group_idx = 0; beam_group_idx < beam_group_num_scalar; ++beam_group_idx) {
    if (beam_group_num_scalar == 1) {
      invokeTopkSoftMax<T, Context, false>(
          dev_ctx, params, beam_group_idx, dev_ctx.stream());
    } else {
      invokeTopkSoftMax<T, Context, true>(
          dev_ctx, params, beam_group_idx, dev_ctx.stream());
    }
  }
  updateBeamSearchParams<T>(params, dev_ctx.stream());

#ifdef DEBUG_BEAM_SEARCH_SOFTMAX
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
  VLOG(2) << "bsf -- next_token: " << *next_tokens;
  VLOG(2) << "bsf -- beam_finished: " << beam_finished;
  VLOG(2) << "bsf -- parent_ids: " << *parent_ids;
  VLOG(2) << "bsf -- seq_lens_out: " << seq_lens;
  VLOG(2) << "bsf -- step_ids_out: " << step_ids;
  VLOG(2) << "bsf -- cache_ids_out: " << beam_cache_ids;
  VLOG(2) << "bsf -- block_tables_out: " << block_tables;
  VLOG(2) << "bsf -- beam_hyps_out: " << beam_hyps;
  VLOG(2) << "bsf -- beam_hyps_score_out: " << beam_hyps_score;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(beam_search_softmax,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::BeamSearchSoftmaxKernel,
                   float) {}  // only supports float
