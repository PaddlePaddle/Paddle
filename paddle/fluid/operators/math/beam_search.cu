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
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {
namespace math {

struct Triple {
  __device__ __forceinline__ Triple() {
    offset = -1;
    id = -1;
    score = -INFINITY;
  }
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

__device__ __forceinline__ int SelectLocalTopBeam(
    Triple* top_beam, const int64_t* pre_ids, const float* pre_scores,
    const int64_t* ids, const float* scores, const int seq_offset_start,
    const int seq_offset_end, const int seq_width, int beam_size, int end_id) {
  for (size_t i = 0; i < beam_size; ++i) {
    top_beam[i].set(-1, -1, -INFINITY);
  }

  int seq_length = seq_offset_end - seq_offset_start;

  int index = seq_offset_start * seq_width;
  int num_items = 0;
  for (int offset = seq_offset_start; offset < seq_offset_end; ++offset) {
    int pre_id = static_cast<int>(pre_ids[offset]);
    if (pre_id == end_id) {
      Triple tmp(offset, end_id, pre_scores[offset]);
      Insert(top_beam, tmp, beam_size);
      num_items = (num_items + 1 > beam_size) ? beam_size : num_items + 1;
      index++;
    } else {
      for (int i = 0; i < seq_width; ++i) {
        Triple tmp(offset, static_cast<int>(ids[index]), scores[index]);
        Insert(top_beam, tmp, beam_size);
        index++;
        num_items = (num_items + 1 > beam_size) ? beam_size : num_items + 1;
      }
    }
  }

  return num_items;
}

__device__ __forceinline__ bool PruneEndBeams(Triple* top_beam,
                                              const int64_t* pre_ids,
                                              const int seq_offset_start,
                                              const int seq_offset_end,
                                              const int end_id, int num_items) {
  bool finish_flag = true;
  int seq_length = seq_offset_end - seq_offset_start;
  for (int i = 0; i < num_items; ++i) {
    int offset = top_beam[i].offset;
    if (top_beam[i].id != end_id ||
        static_cast<int>(pre_ids[offset]) != end_id) {
      finish_flag = false;
      break;
    }
  }
  return finish_flag;
}

__device__ __forceinline__ void Sort(Triple* data, const int num_data) {
  if (num_data <= 1) {
    return;
  }
  for (int i = 0; i < num_data; ++i) {
    for (int j = i + 1; j < num_data; ++j) {
      if (data[j].offset < data[i].offset) {
        Triple tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
      }
    }
  }
}

template <int MaxLength, int NumThreadsPerSeq, int MaxSeqs>
__global__ void BeamSearchKernel(int64_t* selected_ids, float* selected_scores,
                                 const int64_t* pre_ids,
                                 const float* pre_scores, const int64_t* ids,
                                 const float* scores, const size_t* seq_offsets,
                                 const int seq_width, int beam_size, int end_id,
                                 size_t* selected_offsets) {
  __shared__ int shared_mem[MaxSeqs];
  const int tid = threadIdx.x;  // use 1 thread only

  const int seq_id = tid / NumThreadsPerSeq;
  int seq_offset_start = static_cast<int>(seq_offsets[seq_id]);
  int seq_offset_end = static_cast<int>(seq_offsets[seq_id + 1]);

  // if (tid == 0) {
  // printf("seq_offset_start: %d, seq_offset_end: %d\n", seq_offset_start,
  // seq_offset_end);
  // }
  if (tid % NumThreadsPerSeq == 0) {
    Triple top_beam[MaxLength];  // Ensure MaxLength >= beam_size
    int num_items = SelectLocalTopBeam(top_beam, pre_ids, pre_scores, ids,
                                       scores, seq_offset_start, seq_offset_end,
                                       seq_width, beam_size, end_id);

    bool finish_flag = PruneEndBeams(top_beam, pre_ids, seq_offset_start,
                                     seq_offset_end, end_id, num_items);

    int selected_seq_length = finish_flag ? 0 : num_items;
    // [0, MaxSeqs - 1], length of each sequences
    shared_mem[seq_id] = selected_seq_length;
    Sort(top_beam, selected_seq_length);
    __syncthreads();

    int selected_seq_start = 0;
    for (int s = 0; s < seq_id; ++s) {
      selected_seq_start += shared_mem[s];
    }

    if (seq_id == 0) {
      selected_offsets[0] = 0;
    }

    int selected_seq_end = selected_seq_start + selected_seq_length;
    int global_index = selected_seq_start;
    for (int global_offset = selected_seq_start;
         global_offset < selected_seq_end; ++global_offset) {
      int num_items = 0;
      for (int local_index = 0; local_index < selected_seq_length;
           ++local_index) {
        if (top_beam[local_index].offset == global_offset) {
          selected_ids[global_index] =
              static_cast<int64_t>(top_beam[local_index].id);
          selected_scores[global_index] = top_beam[local_index].score;
          global_index++;
        }
      }
      selected_offsets[global_offset + 1] = static_cast<size_t>(global_index);
    }

    // if (seq_id == 0) {
    //   for (int b = 0; b < beam_size; ++b) {
    //     printf("top %u: {offset, %d; id, %d; score, %f}\n", b,
    //     top_beam[b].offset, top_beam[b].id, top_beam[b].score);
    //   }
    //   for (int i = 0; i < MaxSeqs; ++i) {
    //     printf("shared_mem[%d]: %d\n", i, shared_mem[i]);
    //   }
    // }
    // __syncthreads();
    // if (seq_id == 1) {
    //   for (int b = 0; b < beam_size; ++b) {
    //     printf("top %u: {offset, %d; id, %d; score, %f}\n", b,
    //     top_beam[b].offset, top_beam[b].id, top_beam[b].score);
    //   }
    //   for (int i = 0; i < MaxSeqs; ++i) {
    //     printf("shared_mem[%d]: %d\n", i, shared_mem[i]);
    //   }
    // }
  }
}

template <typename T>
class BeamSearchFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::LoDTensor& pre_ids,
                  const framework::LoDTensor& pre_scores,
                  const framework::LoDTensor& ids,
                  const framework::LoDTensor& scores,
                  framework::LoDTensor* selected_ids,
                  framework::LoDTensor* selected_scores, size_t level,
                  size_t beam_size, int end_id) {
    auto abs_lod = framework::ToAbsOffset(ids.lod());
    const size_t* seq_offsets = abs_lod[level].CUDAData(context.GetPlace());

    const int64_t* pre_ids_data = pre_ids.data<int64_t>();
    const float* pre_scores_data = pre_scores.data<float>();
    const int64_t* ids_data = ids.data<int64_t>();
    const float* scores_data = scores.data<float>();

    const size_t num_seqs = abs_lod[level].size() - 1;
    size_t seq_width = 1;
    for (int i = 1; i < ids.dims().size(); i++) {
      seq_width *= ids.dims()[i];
    }

    selected_ids->Resize({static_cast<int64_t>(num_seqs * beam_size), 1});
    int64_t* selected_ids_data =
        selected_ids->mutable_data<int64_t>(context.GetPlace());

    selected_scores->Resize({static_cast<int64_t>(num_seqs * beam_size), 1});
    float* selected_scores_data =
        selected_scores->mutable_data<float>(context.GetPlace());

    // LOG(INFO) << "abs_lod: " << abs_lod;
    // LOG(INFO) << "num_seqs: " << num_seqs << ", seq_width: " << seq_width;
    framework::LoD* selected_lod = selected_ids->mutable_lod();
    selected_lod->resize(2);
    if ((*selected_lod)[1].size() != (num_seqs * beam_size + 1)) {
      (*selected_lod)[1].resize(num_seqs * beam_size + 1);
    }
    (*selected_lod)[0].assign(abs_lod[level].begin(), abs_lod[level].end());
    size_t* selected_offsets =
        (*selected_lod)[1].CUDAMutableData(context.GetPlace());

    if (num_seqs <= 4) {
      // Use onoly 1 block
      BeamSearchKernel<4, 32, 4><<<1, num_seqs * 32, 1024, context.stream()>>>(
          selected_ids_data, selected_scores_data, pre_ids_data,
          pre_scores_data, ids_data, scores_data, seq_offsets,
          static_cast<int>(seq_width), static_cast<int>(beam_size), end_id,
          selected_offsets);
    } else {
      LOG(FATAL) << "Not implemented.";
    }

    if (!framework::CheckLoD(lod)) {
      PADDLE_THROW("lod %s is not right", framework::LoDToString(lod));
    }
    selected_scores->set_lod(*selected_lod);
  }
};

template class BeamSearchFunctor<platform::CUDADeviceContext, int>;
template class BeamSearchFunctor<platform::CUDADeviceContext, int64_t>;
template class BeamSearchFunctor<platform::CUDADeviceContext, float>;
template class BeamSearchFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
