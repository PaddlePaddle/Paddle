/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/beam_search.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

namespace paddle {
namespace operators {
namespace math {

struct Triple {
  __device__ __forceinline__ Triple() {}
  __device__ __forceinline__ Triple(int o, int i, float s)
      : offset(o), id(i), score(s) {}

  __device__ __forceinline__ void set(int o, int i, float s) {
    offset = o;
    id = i;
    score = s;
  }

  __device__ __forceinline__ void operator=(const Triple& in) {
    offset = in.offset;
    id = in.id;
    score = in.score;
  }

  __device__ __forceinline__ bool operator<(const float s) const {
    return score < s;
  }

  __device__ __forceinline__ bool operator<(const Triple& in) const {
    return (score < in.score) || ((score == in.score) && (offset < in.offset));
  }

  int offset;
  int id;
  float score;
};

__device__ __forceinline__ void Insert(Triple* top_beam, const Triple& p,
                                       int beam_size) {
  if (p < top_beam[beam_size - 1]) {
    return;
  }
  for (int k = beam_size - 2; k >= 0; --k) {
    if (top_beam[k] < p) {
      top_beam[k + 1] = top_beam[k];
    } else {
      top_beam[k + 1] = p;
      return;
    }
  }
  top_beam[0] = p;
}

template <int MaxThreadsPerSeq, bool IsAccumulated = true>
__device__ __forceinline__ int SelectTopBeam(
    Triple* top_beam, const int64_t* pre_ids, const float* pre_scores,
    const int64_t* ids, const float* scores, const int seq_offset_start,
    const int seq_offset_end, const int seq_width, int beam_size, int end_id,
    int used_threads) {
  // top_beam is shared memory
  const int tid = threadIdx.x;
  const int tid_of_seq = threadIdx.x % MaxThreadsPerSeq;

  int num_used_threads = used_threads;

  Triple* top_beam_local = top_beam + tid * beam_size;
  if (tid_of_seq < num_used_threads) {
    for (int i = 0; i < beam_size; ++i) {
      top_beam_local[i].set(-1, -1, -INFINITY);
    }

    for (int offset = seq_offset_start; offset < seq_offset_end; ++offset) {
      int pre_id = static_cast<int>(pre_ids[offset]);
      if (pre_id == end_id) {
        if (tid_of_seq == 0) {
          Triple tmp(offset, end_id, pre_scores[offset]);
          Insert(top_beam_local, tmp, beam_size);
        }
      } else {
        int index = offset * seq_width + tid_of_seq;
        if (!IsAccumulated) {
          float pre_score = pre_scores[offset];
          for (int i = tid_of_seq; i < seq_width; i += num_used_threads) {
            float score = pre_score + __logf(scores[index]);
            int id = ids ? static_cast<int>(ids[index]) : i;
            Triple tmp(offset, id, score);
            Insert(top_beam_local, tmp, beam_size);
            index += num_used_threads;
          }
        } else {
          for (int i = tid_of_seq; i < seq_width; i += num_used_threads) {
            int id = ids ? static_cast<int>(ids[index]) : i;
            float score = scores[index];
            Triple tmp(offset, id, score);
            Insert(top_beam_local, tmp, beam_size);
            index += num_used_threads;
          }
        }
      }
    }
  }

  while (num_used_threads > 1) {
    if (num_used_threads > 16) {
      __syncthreads();
    }

    if ((num_used_threads & 0x1) != 0) {
      // If num_used_threads is a odd number, merge local top_beam of thread 0
      // and num_used_threads - 1
      if (tid_of_seq == 0) {
        int index_in_sh = (num_used_threads - 1 + tid) * beam_size;
        for (int i = 0; i < beam_size; i++) {
          Insert(top_beam_local, top_beam[index_in_sh], beam_size);
          index_in_sh++;
        }
      }
    }

    num_used_threads = num_used_threads >> 1;
    if (tid_of_seq < num_used_threads) {
      int index_in_sh = (num_used_threads + tid) * beam_size;
      for (int i = 0; i < beam_size; i++) {
        Insert(top_beam_local, top_beam[index_in_sh], beam_size);
        index_in_sh++;
      }
    }
  }

  if (tid_of_seq == 0) {
    int num_items = 0;
    for (int i = 0; i < beam_size; ++i) {
      num_items =
          (top_beam_local[i].score > -INFINITY) ? num_items + 1 : num_items;
    }
    return num_items;
  }

  return 0;
}

__device__ __forceinline__ bool PruneEndBeams(Triple* top_beam_local,
                                              const int64_t* pre_ids,
                                              const int end_id, int num_items) {
  bool finish_flag = true;
  for (int i = 0; i < num_items; ++i) {
    int offset = top_beam_local[i].offset;
    if (top_beam_local[i].id != end_id ||
        static_cast<int>(pre_ids[offset]) != end_id) {
      finish_flag = false;
      break;
    }
  }
  return finish_flag;
}

template <bool ReturnParentIdx = false>
__device__ __forceinline__ void WriteBack(
    int64_t* selected_ids, float* selected_scores, int* parent_idx,
    size_t* selected_offsets, Triple* top_beam_local,
    const int seq_offset_start, const int seq_offset_end,
    const int selected_seq_start, const int selected_seq_length) {
  const int tid = threadIdx.x;  // use 1 thread only for each sequence
  int global_index = selected_seq_start;
  for (int global_offset = seq_offset_start; global_offset < seq_offset_end;
       ++global_offset) {
    for (int local_index = 0; local_index < selected_seq_length;
         ++local_index) {
      if (top_beam_local[local_index].offset == global_offset) {
        selected_ids[global_index] =
            static_cast<int64_t>(top_beam_local[local_index].id);
        selected_scores[global_index] = top_beam_local[local_index].score;
        if (ReturnParentIdx) {
          parent_idx[global_index] = static_cast<int>(global_offset);
        }
        global_index++;
      }
    }
    selected_offsets[global_offset + 1] = static_cast<size_t>(global_index);
  }
}

template <int MaxLength, int MaxThreadsPerSeq, int MaxSeqs>
__device__ void BeamSearchDetails(
    int64_t* selected_ids, float* selected_scores, int* parent_idx,
    size_t* selected_offsets, const int64_t* pre_ids, const float* pre_scores,
    const int64_t* ids, const float* scores, const int seq_offset_start,
    const int seq_offset_end, const int seq_width, int beam_size, int end_id,
    bool is_accumulated, int num_used_threads) {
  __shared__ Triple top_beam[MaxLength];

  int num_items = 0;
  if (is_accumulated) {
    num_items = SelectTopBeam<MaxThreadsPerSeq, true>(
        top_beam, pre_ids, pre_scores, ids, scores, seq_offset_start,
        seq_offset_end, seq_width, beam_size, end_id, num_used_threads);
  } else {
    num_items = SelectTopBeam<MaxThreadsPerSeq, false>(
        top_beam, pre_ids, pre_scores, ids, scores, seq_offset_start,
        seq_offset_end, seq_width, beam_size, end_id, num_used_threads);
  }

  const int tid = threadIdx.x;  // use 1 thread only for each sequence
  const int tid_of_seq = tid % MaxThreadsPerSeq;
  if (tid_of_seq == 0) {
    // Use 1 thread for each sequence.
    Triple* top_beam_local = top_beam + tid * beam_size;
    bool finish_flag =
        PruneEndBeams(top_beam_local, pre_ids, end_id, num_items);

    int selected_seq_start = 0;
    int selected_seq_length = finish_flag ? 0 : num_items;

    if (MaxSeqs > 1) {
      const int seq_id = (MaxSeqs > 1) ? tid / MaxThreadsPerSeq : tid;
      __shared__ int shared_mem[MaxSeqs];

      // [0, MaxSeqs - 1], length of each sequences
      shared_mem[seq_id] = selected_seq_length;
      __syncthreads();

      for (int s = 0; s < seq_id; ++s) {
        selected_seq_start += shared_mem[s];
      }

      if (seq_id == 0) {
        selected_offsets[0] = 0;
      }
    } else {
      selected_offsets[0] = 0;
    }

    if (parent_idx) {
      WriteBack<true>(selected_ids, selected_scores, parent_idx,
                      selected_offsets, top_beam_local, seq_offset_start,
                      seq_offset_end, selected_seq_start, selected_seq_length);
    } else {
      WriteBack<false>(selected_ids, selected_scores, parent_idx,
                       selected_offsets, top_beam_local, seq_offset_start,
                       seq_offset_end, selected_seq_start, selected_seq_length);
    }
  }
}

template <int MaxLength, int MaxThreadsPerSeq, int MaxSeqs>
__global__ void BeamSearchKernel(int64_t* selected_ids, float* selected_scores,
                                 int* parent_idx, size_t* selected_offsets,
                                 const int64_t* pre_ids,
                                 const float* pre_scores, const int64_t* ids,
                                 const float* scores, const size_t* seq_offsets,
                                 const int num_seqs, const int seq_width,
                                 int beam_size, int end_id, bool is_accumulated,
                                 int num_used_threads) {
  const int tid = threadIdx.x;
  const int seq_id = (MaxSeqs > 1) ? tid / MaxThreadsPerSeq : tid;

  int seq_offset_start = static_cast<int>(seq_offsets[seq_id]);
  int seq_offset_end = static_cast<int>(seq_offsets[seq_id + 1]);

  BeamSearchDetails<MaxLength, MaxThreadsPerSeq, MaxSeqs>(
      selected_ids, selected_scores, parent_idx, selected_offsets, pre_ids,
      pre_scores, ids, scores, seq_offset_start, seq_offset_end, seq_width,
      beam_size, end_id, is_accumulated, num_used_threads);
}

template <int MaxLength, int MaxThreadsPerSeq>
__global__ void BeamSearchKernelSingle(
    int64_t* selected_ids, float* selected_scores, int* parent_idx,
    size_t* selected_offsets, const int64_t* pre_ids, const float* pre_scores,
    const int64_t* ids, const float* scores, const int seq_length,
    const int seq_width, int beam_size, int end_id, bool is_accumulated,
    int num_used_threads) {
  const int seq_offset_start = 0;
  const int seq_offset_end = seq_length;

  BeamSearchDetails<MaxLength, MaxThreadsPerSeq, 1>(
      selected_ids, selected_scores, parent_idx, selected_offsets, pre_ids,
      pre_scores, ids, scores, seq_offset_start, seq_offset_end, seq_width,
      beam_size, end_id, is_accumulated, num_used_threads);
}

static inline int GetNumUsedThreads(const int max_threads_per_seq,
                                    const int seq_width, int beam_size) {
  int num_used_threads = (seq_width + beam_size - 1) / beam_size;
  num_used_threads = max_threads_per_seq < num_used_threads
                         ? max_threads_per_seq
                         : num_used_threads;

  num_used_threads =
      num_used_threads > 32
          ? (num_used_threads >> 5) << 5
          : (num_used_threads > 16
                 ? 32
                 : (num_used_threads > 8
                        ? 16
                        : (num_used_threads > 4
                               ? 8
                               : (num_used_threads > 2 ? 4
                                                       : num_used_threads))));
  return num_used_threads;
}

template <typename T>
class BeamSearchFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::LoDTensor* pre_ids,
                  const framework::LoDTensor* pre_scores,
                  const framework::LoDTensor* ids,
                  const framework::LoDTensor* scores,
                  framework::LoDTensor* selected_ids,
                  framework::LoDTensor* selected_scores,
                  framework::Tensor* parent_idx, size_t level, size_t beam_size,
                  int end_id, bool is_accumulated) {
    auto abs_lod = framework::ToAbsOffset(scores->lod());

    const int64_t* pre_ids_data = pre_ids->data<int64_t>();
    const float* pre_scores_data = pre_scores->data<float>();
    const int64_t* ids_data = ids ? ids->data<int64_t>() : nullptr;
    const float* scores_data = scores->data<float>();

    const size_t num_seqs = abs_lod[level].size() - 1;
    size_t seq_width = 1;
    for (int i = 1; i < scores->dims().size(); i++) {
      seq_width *= scores->dims()[i];
    }

    // Reserve a big enough memory.
    auto selected_dims =
        framework::make_ddim({static_cast<int64_t>(num_seqs * beam_size), 1});
    int64_t* selected_ids_data =
        selected_ids->mutable_data<int64_t>(selected_dims, context.GetPlace());
    float* selected_scores_data =
        selected_scores->mutable_data<float>(selected_dims, context.GetPlace());
    int* parent_idx_data =
        parent_idx
            ? parent_idx->mutable_data<int>(
                  {static_cast<int64_t>(num_seqs * beam_size)},
                  context.GetPlace())
            : nullptr;

    framework::LoD selected_lod(2);
    selected_lod[0].assign(abs_lod[level].begin(), abs_lod[level].end());
    selected_lod[1].resize(scores->dims()[0] + 1);
    size_t* selected_offsets =
        selected_lod[1].CUDAMutableData(context.GetPlace());

    if (num_seqs == 1) {
      const int seq_length = static_cast<int>(abs_lod[level][1]);
      const int kMaxThreadsPerSeq = 1024;
      int num_used_threads =
          GetNumUsedThreads(kMaxThreadsPerSeq, static_cast<int>(seq_width),
                            static_cast<int>(beam_size));
      switch (platform::RoundToPowerOfTwo(beam_size * seq_width)) {
        CUDA_LAUNCH_KERNEL_HELPER(
            BeamSearchKernelSingle<kPowerOfTwoDim, kMaxThreadsPerSeq><<<
                1, kMaxThreadsPerSeq, 0, context.stream()>>>(
                selected_ids_data, selected_scores_data, parent_idx_data,
                selected_offsets, pre_ids_data, pre_scores_data, ids_data,
                scores_data, seq_length, static_cast<int>(seq_width),
                static_cast<int>(beam_size), static_cast<int>(end_id),
                is_accumulated, num_used_threads));
      }
    } else if (num_seqs <= 4) {
      const size_t* seq_offsets = abs_lod[level].CUDAData(context.GetPlace());
      // Use only 1 block
      const int kMaxThreadsPerSeq = 32;
      const int kMaxSeqs = 4;
      int num_used_threads =
          GetNumUsedThreads(kMaxThreadsPerSeq, static_cast<int>(seq_width),
                            static_cast<int>(beam_size));
      switch (platform::RoundToPowerOfTwo(beam_size * num_seqs * 32)) {
        CUDA_LAUNCH_KERNEL_HELPER(
            BeamSearchKernel<kPowerOfTwoDim, kMaxThreadsPerSeq, kMaxSeqs><<<
                1, num_seqs * kMaxThreadsPerSeq, 0, context.stream()>>>(
                selected_ids_data, selected_scores_data, parent_idx_data,
                selected_offsets, pre_ids_data, pre_scores_data, ids_data,
                scores_data, seq_offsets, static_cast<int>(num_seqs),
                static_cast<int>(seq_width), static_cast<int>(beam_size),
                end_id, is_accumulated, num_used_threads));
      }
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Not implemented other number of sequences yet."));
    }

    context.Wait();
    if (!framework::CheckLoD(selected_lod)) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "lod %s is not right in"
          " beam_search, please check your code.",
          framework::LoDToString(selected_lod)));
    }

    selected_ids->set_lod(selected_lod);
    selected_scores->set_lod(selected_lod);
    if (selected_lod[1].back() < num_seqs * beam_size) {
      auto final_selected_dims = framework::make_ddim(
          {static_cast<int64_t>(selected_lod[1].back()), 1});
      selected_ids->Resize(final_selected_dims);
      selected_scores->Resize(final_selected_dims);
      if (parent_idx) {
        parent_idx->Resize({static_cast<int64_t>(selected_lod[1].back())});
      }
    }
  }
};

template class BeamSearchFunctor<platform::CUDADeviceContext, int>;
template class BeamSearchFunctor<platform::CUDADeviceContext, int64_t>;
template class BeamSearchFunctor<platform::CUDADeviceContext, float>;
template class BeamSearchFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
